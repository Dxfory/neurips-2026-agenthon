"""Static admission for one generated file; never executes code or touches files.

``ready`` means a syntactically executable candidate with a visible module call,
not successful execution, delivery, or mathematical correctness. The caller owns
code/output snapshots and rollback. A bundle replaces whole top-level bindings;
ambiguous statement edits require an explicit whole-file replacement.
"""
import ast
import builtins
import codeop
from dataclasses import dataclass
import hashlib
from pathlib import PurePosixPath
import re
import symtable
import warnings

from . import continuation


_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_IMPORTS = (ast.Import, ast.ImportFrom)


def _valid_path(path):
    return (isinstance(path, str) and bool(path) and "\\" not in path
            and PurePosixPath(path).name == path and path not in (".", "..")
            and path.endswith(".py"))


def _parse(source):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # AST parsing alone accepts, for example, a module-level return.
        compile(source, "<generated>", "exec", dont_inherit=True)
        return ast.parse(source)


class FormatError(ValueError):
    """A reply whose program cannot be extracted; ``detail`` names the exact problem for feedback."""

    def __init__(self, reason, detail=""):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


@dataclass(frozen=True)
class Extraction:
    source: str      # the selected code, newline-terminated ("" when empty)
    cut: bool        # the code ends where a length reply was cut (only then may a cut tail be salvaged)
    shape: str       # raw | fenced (one clean block) | tolerant (prose, stray fences or other blocks dropped)


def _python_tag(tag):
    return re.fullmatch(r"py3?|python3?(?:\.\d+)*", tag) is not None


def _fence_tag(stripped):
    rest = stripped.lstrip("`").strip()
    return rest.split()[0].lower() if rest else ""


def _quote_after(line, quote):
    """The triple-quoted string still open after this line (None when none is), from the one open before it."""
    pos = 0
    while True:
        if quote is None:
            hits = [(at, mark) for at, mark in ((line.find(mark, pos), mark) for mark in ('"""', "'''")) if at >= 0]
            if not hits:
                return None
            pos, quote = min(hits)
        else:
            pos = line.find(quote, pos)
            if pos < 0:
                return quote
            quote = None
        pos += 3


def _fence_lines(lines):
    """Indices of the fence lines. A fence-like line inside a triple-quoted string (a program writing
    markdown) is content, unless that reading leaves a string open at the end, i.e. a stray quote in
    prose or a cut: then every fence-like line counts."""
    plain = [i for i, line in enumerate(lines) if line.strip().startswith("```")]
    fences, quote = [], None
    for i, line in enumerate(lines):
        if quote is None and line.strip().startswith("```"):
            fences.append(i)
        else:
            quote = _quote_after(line, quote)
    return fences if quote is None else plain


def _segments(lines, fences, headers=None):
    """[(tag, lines, part, end)] of the non-blank parts: tag None for text outside fences, "" for an
    untagged fence, else the fence's info word; ``end`` is the index of the part's last line. A bare
    fence line closes the open block. A tagged fence line inside a block opens a new block: an empty
    block before it was a stray fence, a non-empty one was left unclosed by the model. A block still
    open at the end runs to the end of the reply. A repeated header line (``headers``: index -> action)
    ends the part before it and starts the next reply part (``part`` counts them) with the same tag; in
    a reply without any fence line every part is code."""
    headers = headers or {}
    out, current, part, fences = [], [], 0, set(fences)
    tag = "python" if headers and not fences else None
    for index, line in enumerate(lines):
        if index in headers:
            out.append((tag, current, part, index - 1))
            current, part = [], part + 1
            continue
        if index not in fences:
            current.append(line)
            continue
        word = _fence_tag(line.strip())
        out.append((tag, current, part, index - 1))
        if tag is not None and not word:
            current, tag = [], None
        else:
            current, tag = [], word
    out.append((tag, current, part, len(lines) - 1))
    return [item for item in out if any(line.strip() for line in item[1])]


def _program_like(source):
    """Parses and holds a statement, not only a value such as an output example."""
    try:
        tree = _parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError):
        return False
    return any(not isinstance(node, (ast.Expr, ast.AnnAssign)) or isinstance(node.value, ast.Call)
               for node in tree.body)


def _status(source, cut):
    """'ok' when the code compiles; 'cut' when it does not but ends at a length cut ProgramVersion can
    salvage (only for code that ends the reply, ``cut``); else 'broken'."""
    try:
        _parse(source)
    except SyntaxError as error:
        return "cut" if cut and _cutoff(source, error) else "broken"
    except (ValueError, RecursionError, MemoryError, OverflowError):
        return "broken"
    return "ok"


# Calls that write a file. A block that uses the selected block's names and writes is program, not a usage example.
_WRITE_CALLS = frozenset({"dump", "to_csv", "to_json", "to_parquet", "to_excel", "to_pickle", "to_feather", "to_hdf",
                          "savefig", "save", "savez", "savez_compressed", "savetxt", "write_text", "write_bytes",
                          "write", "writelines", "writerow", "writerows"})


def _mode_string(node):
    return (isinstance(node, ast.Constant) and isinstance(node.value, str)
            and re.fullmatch(r"[rwxabtU+]{1,4}", node.value) is not None and bool(set(node.value) & set("wax+")))


def _writes(source):
    """Whether the code (its complete top-level statements) calls something that writes a file:
    open(..., 'w'/'a'/'x'/'+'), Path.open('w'), json.dump, DataFrame.to_csv, savefig, np.save, .write ...
    Writes to sys.stdout / sys.stderr do not count."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(_complete(source))
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
        if isinstance(func, ast.Attribute):
            root = func.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id == "sys":
                continue
        if name in _WRITE_CALLS:
            return True
        if name == "open":
            modes = [kw.value for kw in node.keywords if kw.arg == "mode"]
            modes += node.args[1:2] if isinstance(func, ast.Name) else node.args[:2]
            if any(_mode_string(mode) for mode in modes):
                return True
    return False


def module_names(source):
    """(names bound at module scope, names read from module scope but bound nowhere), or None when
    that is unknowable (a star import, a parse or scope error). Builtins and dunders are not free."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(source)
            top = symtable.symtable(source, "<reply>", "exec")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    if any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names) for n in ast.walk(tree)):
        return None
    bound, read, todo = set(), set(), [top]
    while todo:
        table = todo.pop()
        todo.extend(table.get_children())
        for symbol in table.get_symbols():
            module = table is top
            if (module or symbol.is_declared_global()) and (
                    symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()):
                bound.add(symbol.get_name())
            if symbol.is_referenced() and (module or symbol.is_global()):
                read.add(symbol.get_name())
    return bound, {name for name in read - bound
                   if not hasattr(builtins, name) and not (name.startswith("__") and name.endswith("__"))}


def _defined(source):
    try:
        return {node.name for node in ast.parse(source).body if isinstance(node, _DEFS)}
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return set()


def _names(values, limit=5):
    values = sorted(values)
    return ", ".join(values[:limit]) + (", ..." if len(values) > limit else "")


def _complete(source):
    """The whole top-level statements that compile (all of a block that compiles), for name analysis."""
    return continuation.verified_prefix(source, True)[0]


def _dedupe(items):
    return list({source: (source, cut) for source, cut in items}.values())


def _whole(source, mode, known):
    """A program by itself: holds a statement, reads no module name it never binds (that ``known`` lacks)
    and, for a REPLACE, has an entry call. A usage line, an output value or bare definitions are not."""
    names = module_names(source)
    return (_program_like(source) and names is not None and not names[1] - known
            and (mode == "bundle" or any(_has_call(node) for node in _parse(source).body)))


_HEADER_LINE = re.compile(r"[ \t]*(REPLACE|BUNDLE)[ \t]+[0-9a-f]{64}[ \t]*")


def extract_code(content, finish="stop", *, mode="replace", expected_files=(), program=""):
    """Select the one program in a code reply by a fixed rule; never join blocks.

    0. A line repeating the action header (REPLACE/BUNDLE and a full version) is never code: like a fence
       line it ends the block or text before it, and what follows is another reply part, read by the rules
       below together with every other block (in a reply without any fence line each part is code). The
       choice must sit under the reply's own action (a BUNDLE on an empty program is a REPLACE); under a
       header naming the other action it is ambiguous.
    1. Text that compiles as Python as it stands, or text without any fence line, is the code. So is the
       body of one python (or untagged) fence that opens the reply, with at most its closing fence after.
    2. Otherwise fences are read line by line: a stray or duplicated bare fence opens or closes an empty
       block, which is ignored; a fence-like line inside a triple-quoted string is code. Prose and blocks
       tagged otherwise (json, text, bash ...) are dropped.
    3. Candidates, in strict tiers: python-tagged blocks that compile; only when there is none, the block
       that ends a length reply when it stops at a cut ProgramVersion can salvage; only when neither
       exists, untagged blocks or text outside fences that compile and hold a statement (not a lone value
       such as an output example). Next to the block a length reply was cut in (salvageable or compiling),
       a closed compiling block counts only when it is a whole program by itself (a statement, an entry
       call for a REPLACE, no name read that it never binds or the current program of a BUNDLE lacks), so
       a usage line, an import or an output value never displaces the program being cut. Within a tier a
       closed block ranks before the block a length reply was cut in, then a block naming every expected
       output file, then the longest; two different equally ranked blocks are ambiguous.
    4. A python-tagged block that fails to compile (other than the one a length reply was cut in) and is
       longer than the choice is returned instead, so its own syntax error is reported rather than a
       smaller block, such as a usage line, being run as the whole program.
    5. The choice is refused as a split program when it reads names that only another code block binds;
       for a REPLACE, when it has no entry call and another block uses its names; and when another block
       uses its names and writes a file (or names an expected output it does not), which is the rest of
       the program, not a usage example; for a BUNDLE, also when another code block defines functions or
       classes it lacks, unless that block redefines one of its own (then it is another version of the
       group, not a part). A block of another reply part (after a repeated header line) that redefines one
       of the choice's functions or classes is likewise another version of the reply, not the rest of the
       program. Blocks that fail to compile take part through their complete statements.
    6. With no candidate, the longest python (else untagged) block that fails to compile is returned,
       so its own syntax error, or a genuine cut, is handled by the caller.
    """
    text = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return Extraction("", False, "raw")
    lines = text.split("\n")
    headers = {i: match.group(1).lower() for i, line in enumerate(lines)
               for match in (_HEADER_LINE.fullmatch(line),) if match}
    fences = _fence_lines(lines)
    if not headers and (not fences or (fences[0] != 0 and _status(text + "\n", False) == "ok")):
        return Extraction(text.rstrip() + "\n", finish == "length", "raw")
    opening = _fence_tag(lines[0].strip())
    if (not headers and fences[0] == 0 and (not opening or _python_tag(opening))
            and lines[0].strip().lstrip("`").strip().lower() == opening
            and (fences == [0] or (fences == [0, len(lines) - 1] and lines[-1].strip() == "```"))):
        body = "\n".join(lines[1:-1] if len(fences) == 2 else lines[1:])
        return Extraction(body.rstrip() + "\n" if body.strip() else "", finish == "length", "fenced")
    segments = _segments(lines, fences, headers)
    if not segments:
        return Extraction("", False, "tolerant")  # fence lines only: an empty code block
    python, salvage, broken, failed, other, failed_bare, dropped = [], [], [], [], [], [], []
    # The reply part (and so the header action) each block came from; part 0 follows the reply's own header.
    actions = [None] + [headers[i] for i in sorted(headers)]
    parts = {}
    for tag, body, part, end in segments:
        source = "\n".join(body).strip("\n").rstrip() + "\n"
        parts.setdefault(source, set()).add(part)
        # A block still open where the length reply was cut.
        cut = finish == "length" and tag is not None and end == len(lines) - 1
        if tag is not None and _python_tag(tag):
            status = _status(source, cut)
            if status == "ok":
                python.append((source, cut))
            elif status == "cut":
                salvage.append((source, cut))
            else:
                (failed if cut else broken).append((source, cut))  # failed: unsalvageable cut block
        elif tag in (None, "") and _program_like(source):
            other.append((source, cut))
        elif tag == "" and _status(source, False) != "ok":
            failed_bare.append((source, cut))
            dropped.append("an untagged block")
        else:
            dropped.append("prose" if tag is None else "an untagged value block" if tag == ""
                           else "a ```" + tag + " block")
    python, broken, other = _dedupe(python), _dedupe(broken), _dedupe(other)
    context = module_names(program) if mode == "bundle" and program else None
    known = context[0] if context else set()

    def effective(action):
        action = action or mode
        return "replace" if action == "bundle" and not program else action

    def selected(source, cut):
        # The chosen code must sit under the reply's own action, not under a repeated header naming the other.
        own = effective(None)
        if headers and own not in {effective(actions[part]) for part in parts[source]}:
            raise FormatError("ambiguous_code_blocks", "the selected code follows a repeated "
                              + ("BUNDLE" if own == "replace" else "REPLACE") + " line but the reply's header is "
                              + own.upper() + "; send one header line and one program")
        return Extraction(source, cut, "tolerant")
    # A usage line, an import or an output value never displaces the program a length reply was cut in.
    complete = ([item for item in python if item[1] or _whole(item[0], mode, known)]
                if salvage or any(cut for _, cut in python) else python)
    pool = complete or salvage or other
    if not pool:
        fallback = broken + failed or failed_bare
        if fallback:
            return selected(*max(fallback, key=lambda item: len(item[0].strip())))
        raise FormatError("no_code_block", "no ```python block was found (the reply had "
                          + (", ".join(sorted(set(dropped))) or "nothing else") + ")")
    files = [PurePosixPath(name).name for name in expected_files if PurePosixPath(name).name]

    def rank(item):
        return (not item[1], bool(files) and all(name in item[0] for name in files), len(item[0].strip()))
    pool = sorted(pool, key=rank, reverse=True)
    source, cut = pool[0]
    # A longer python block with a genuine syntax error is the program the reply meant: report its error.
    longest = max(broken, key=lambda item: len(item[0].strip()), default=None)
    if longest is not None and len(longest[0].strip()) > len(source.strip()):
        return selected(*longest)
    if len(pool) > 1 and rank(pool[1]) == rank(pool[0]):
        raise FormatError("ambiguous_code_blocks", f"{len(pool)} different code blocks of equal length")
    others = [item[0] for item in _dedupe(python + salvage + broken + failed + other) if item[0] != source]
    chosen = _complete(source)
    names = module_names(chosen) if others else None
    if names is not None:
        split = f"the program is split over {len(others) + 1} code blocks: "
        found = [(item, result) for item in others for result in (module_names(_complete(item)),) if result]
        needed = (names[1] & set().union(*(result[0] for _, result in found))) - known
        if needed:
            raise FormatError("split_code_blocks", split + "the selected block uses names defined only in "
                              "another block (" + _names(needed) + ")")
        mine = _defined(chosen)
        # A block of another reply part that redefines one of the choice's definitions is another version.
        versions = {item for item, _ in found if not parts[item] & parts[source] and _defined(_complete(item)) & mine}
        users = [(item, names[0] & result[1]) for item, result in found
                 if names[0] & result[1] and item not in versions]
        used = set().union(*(names_used for _, names_used in users))
        if mode == "replace" and used and not any(_has_call(node) for node in _parse(chosen).body):
            raise FormatError("split_code_blocks", split + "the selected block has no entry call and another "
                              "block uses its names (" + _names(used) + ")")
        for item, names_used in users:
            outputs = [name for name in files if name in item and name not in source]
            if outputs or _writes(item):
                raise FormatError("split_code_blocks", split + "another block uses the selected block's names ("
                                  + _names(names_used) + ") and writes "
                                  + (_names(outputs) if outputs else "a file") + " itself")
        if mode == "bundle":
            # A block that redefines one of the choice's functions or classes is another version of it, not a part.
            groups = [defs for defs in (_defined(_complete(item)) for item in others) if not defs & mine]
            missing = set().union(*groups) - mine
            if missing:
                raise FormatError("split_code_blocks", f"the BUNDLE is spread over {len(others) + 1} code "
                                  "blocks: other blocks define " + _names(missing))
    return selected(source, cut)


def _source(content, finish="stop", **kwargs):
    """The code of one reply (see extract_code); raises FormatError when none can be extracted."""
    return extract_code(content, finish, **kwargs).source


def _cutoff(source, error):
    """Only explicit length replies with a plausible Python EOF get salvage."""
    try:
        if codeop.compile_command(source, symbol="exec") is None:
            return True
    except (SyntaxError, ValueError, OverflowError):
        pass
    if any(mark in error.msg for mark in ("was never closed", "unterminated string literal",
                                         "unterminated triple-quoted string literal")):
        return True
    last = source.rstrip().splitlines()[-1] if source.strip() else ""
    if error.lineno != len(source.rstrip().splitlines()):
        return False
    return bool(re.search(r"(?:=|,|\+|-|\*|/|\b(?:in|as|and|or))\s*$", last)
                or re.fullmatch(r"\s*(?:async\s+)?(?:def|class)\s+[A-Za-z_]\w*", last))


_COMPOUND = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.If, ast.For, ast.AsyncFor, ast.While,
             ast.With, ast.AsyncWith, ast.Try, ast.TryStar, ast.Match)
# A line that cannot begin a top-level statement continues the one before it (as verified_prefix reads lines).
_CONTINUES = re.compile(r"(?:\)|\]|\}|else\b|elif\b|except\b|finally\b|case\b|#)")


def _still_open(content):
    """Whether the reply ends inside its code: its last non-blank line is not a fence line."""
    nonblank = [line for line in content.replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]
    return bool(nonblank) and not nonblank[-1].strip().startswith("```")


def _closing_fence_cut(content, finish):
    """A length reply cut inside its closing fence (a last line of one or two backticks after an opening
    fence) is a closed block: the fragment becomes the closing fence."""
    lines = content.replace("\r\n", "\n").replace("\r", "\n").rstrip().split("\n")
    if (finish == "length" and len(lines) > 1 and re.fullmatch(r"[ \t]*`{1,2}[ \t]*", lines[-1])
            and any(line.strip().startswith("```") for line in lines[:-1])):
        return "\n".join(lines[:-1] + ["```"])
    return content


def _open_cut(content, source):
    """(prefix, tail) of code that compiles but ends where a length reply was cut inside its still-open
    block. The partial last line is unfinished, and so is a compound last statement (def, class, if, for,
    while, with, try, match) that reaches the cut; a complete simple last line stays. Both go to the tail,
    so the obligation is recorded. (source, "") when the block was closed before the cut."""
    if not _still_open(content):
        return source, ""
    text = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = source.rstrip("\n").split("\n")
    partial = ""
    if text.split("\n")[-1].strip():  # the reply stops inside its last line
        partial, lines = lines[-1], lines[:-1]
    kept = "\n".join(lines).rstrip() + "\n" if any(line.strip() for line in lines) else ""
    prefix, tail = continuation.verified_prefix(kept, True)
    # A partial line that starts a new top-level statement means the statement before it was complete.
    # (After a line ending in a backslash the kept lines never compile, so the tail is already set.)
    reaches_cut = not partial or partial[:1].isspace() or _CONTINUES.match(partial) is not None
    body = _parse(prefix).body if prefix.strip() else []
    if not tail.strip() and reaches_cut and body and isinstance(body[-1], _COMPOUND):
        prefix, tail = continuation.verified_prefix(prefix, False)
    tail = "\n".join(part for part in (tail.rstrip("\n"), partial) if part.strip())
    return prefix, tail


def _code_lines(tail):
    """The tail from its first line of code on; '' when it holds only comments and blank lines."""
    lines = tail.split("\n")
    first = next((i for i, line in enumerate(lines) if line.strip() and not line.strip().startswith("#")), None)
    return "" if first is None else "\n".join(lines[first:])


_GUARD = "if __name__"  # the obligation of a dropped __main__ guard; never a binding name
# The header line of a __main__ guard, also when the cut fell inside it (`if __name__ == '__ma`).
_GUARD_LINE = re.compile(r"\s*if\s+\(?\s*(?:__name__\b|['\"]__main__['\"])")


def _tail_name(tail):
    # The obligation identifies a whole binding, never a literal continuation; a dropped __main__ guard is
    # owed as the entry (_GUARD) and other unnamed top-level work as the module (None).
    tail = _code_lines(tail)
    match = re.match(r"\s*(?:@[^\n]+\n\s*)*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)", tail)
    if not match:
        match = re.match(r"\s*([A-Za-z_]\w*)\s*(?::[^=\n]+)?=(?!=)", tail)
    if not match:
        return _GUARD if _GUARD_LINE.match(tail) else None
    return match.group(1)


def _bindings(node):
    if isinstance(node, _DEFS):
        return (node.name,)
    if isinstance(node, ast.ImportFrom) and node.module == "__future__":
        return ()
    if isinstance(node, _IMPORTS):
        if any(alias.name == "*" for alias in node.names):
            return ()
        return tuple(sorted(alias.asname or (alias.name.split(".")[0] if isinstance(node, ast.Import)
                                            else alias.name) for alias in node.names))
    targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
    if targets and all(isinstance(target, ast.Name) for target in targets):
        return tuple(sorted(target.id for target in targets))
    return ()


def _key(node):
    if isinstance(node, ast.ImportFrom) and node.module == "__future__":
        return ("future_import", tuple(alias.name for alias in node.names))
    names = _bindings(node)
    if names:
        return ("bindings", names)
    if isinstance(node, ast.If) and continuation.is_main_guard(node):
        return ("entry_guard",)
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
        return ("module_doc",)
    return None


def _noop(node):
    return isinstance(node, ast.Pass) or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                                        and node.value.value is Ellipsis)


def _placeholder(node):
    return continuation.placeholder(node) or all(_noop(s) or (
        isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))
        for s in node.body)


def _has_call(node):
    """Visible execution, without inferring task work or dynamic dependencies."""
    if isinstance(node, _DEFS + (ast.Lambda,)):
        return False
    if isinstance(node, ast.Call):
        return True
    if isinstance(node, (ast.If, ast.While)) and isinstance(node.test, ast.Constant):
        selected = node.body if node.test.value else node.orelse
        return any(_has_call(child) for child in selected)
    return any(_has_call(child) for child in ast.iter_child_nodes(node))


def _names_only(target):
    return isinstance(target, ast.Name) or (isinstance(target, (ast.Tuple, ast.List))
                                            and all(_names_only(item) for item in target.elts))


def _entry(node, defined):
    """Visible top-level work: a statement with a call (an expression, a __main__ guard or another compound
    statement, a raise, an item or attribute store ...), except that binding plain names counts only when
    it calls a function or class this program defines (``results = main()``), not a library
    (``OUT = Path(...)``, ``rng = np.random.default_rng(0)``)."""
    if isinstance(node, _DEFS + _IMPORTS):
        return False
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if all(_names_only(target) for target in targets):
            return node.value is not None and any(
                isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in defined
                for call in ast.walk(node.value))
    return _has_call(node)


def _has_entry(nodes, defined):
    return any(_entry(node, defined) for node in nodes)


def _resupplies(node, defined):
    """Top-level work that stands in for dropped top-level statements: a compound statement with a call (a
    __main__ guard, a loop, a with ...) or a statement that uses one of the program's own functions or
    classes. A bare library call (print, a display setting, makedirs), a library binding, an import or a
    definition does not."""
    if isinstance(node, _DEFS + _IMPORTS):
        return False
    if isinstance(node, _COMPOUND):
        return _has_call(node)
    return any(isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load) and item.id in defined
               for item in ast.walk(node))


def _runs_guard(nodes):
    """A complete __main__ guard that calls something: the program's entry."""
    return any(isinstance(node, ast.If) and continuation.is_main_guard(node) and _has_call(node) for node in nodes)


class _MergeError(ValueError):
    """A BUNDLE that cannot be merged unambiguously; ``detail`` names the statements for feedback."""

    def __init__(self, reason, detail=""):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


def _label(node, limit=60):
    """Short model-facing name of a top-level statement: its bound names, else its first source line."""
    names = _bindings(node)
    if names:
        return ", ".join(names)[:limit]
    if isinstance(node, ast.If) and continuation.is_main_guard(node):
        return "the __main__ guard"
    try:
        text = ast.unparse(node).split("\n", 1)[0]
    except (ValueError, RecursionError):
        text = type(node).__name__
    return "`" + text[:limit] + "`"


def _repeated(nodes):
    """Names that these statements bind more than once; an identical repeated import counts once."""
    names, imports = [], set()
    for node in nodes:
        if isinstance(node, _IMPORTS):
            dump = ast.dump(node)
            if dump in imports:
                continue
            imports.add(dump)
        names.extend(_bindings(node))
    return {name for name in names if names.count(name) > 1}


def _double_entries(nodes):
    """The program's own function or class that a __main__ guard runs as its only own statement-level call
    while a bare top-level statement calls it too, i.e. the entry would run twice. A guard that runs more
    (``setup(); main()`` beside a bare ``setup()``) is a new entry, and a nested call (``print(f())``) is
    not a statement-level call."""
    defined = {node.name for node in nodes if isinstance(node, _DEFS)}

    def direct(statement):
        call = statement.value if isinstance(statement, ast.Expr) else None
        return (call.func.id if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id in defined else None)
    bare = {direct(node) for node in nodes} - {None}
    doubled = set()
    for node in nodes:
        if isinstance(node, ast.If) and continuation.is_main_guard(node):
            runs = {direct(statement) for statement in node.body} - {None}
            if len(runs) == 1 and runs <= bare:
                doubled |= runs
    return doubled


def _eager_names(node, annotations):
    """Names a def or class statement reads while it runs (not when it is called): decorators, defaults,
    bases and class keywords, a class body outside its methods and, unless ``from __future__ import
    annotations`` defers them (``annotations`` False), the annotations Python 3.13 evaluates eagerly."""
    exprs = list(node.decorator_list)
    if isinstance(node, ast.ClassDef):
        exprs += node.bases + [keyword.value for keyword in node.keywords]
        for statement in node.body:
            if isinstance(statement, _DEFS):
                exprs.extend(ast.Name(id=name, ctx=ast.Load()) for name in _eager_names(statement, annotations))
            else:
                exprs.append(statement)
    else:
        args = node.args
        exprs += args.defaults + [value for value in args.kw_defaults if value is not None]
        if annotations:
            every = args.posonlyargs + args.args + args.kwonlyargs + [args.vararg, args.kwarg]
            exprs += [arg.annotation for arg in every if arg is not None and arg.annotation is not None]
            exprs += [node.returns] if node.returns is not None else []
    return {item.id for expr in exprs for item in ast.walk(expr)
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)}


def _future_annotations(nodes):
    return any(isinstance(node, ast.ImportFrom) and node.module == "__future__"
               and any(alias.name == "annotations" for alias in node.names) for node in nodes)


def _existing(old, node):
    """Index in ``old`` of the statement ``node`` replaces or repeats, else None."""
    dump = ast.dump(node)
    exact = next((i for i, previous in enumerate(old) if ast.dump(previous) == dump), None)
    if exact is not None:
        return exact
    key = _key(node)
    matching = [i for i, previous in enumerate(old) if key is not None and _key(previous) == key]
    return matching[0] if len(matching) == 1 else None


def _bound(nodes):
    """Names the statements bind at module scope (empty when that cannot be told)."""
    names = module_names(ast.unparse(ast.Module(body=list(nodes), type_ignores=[]))) if nodes else None
    return names[0] if names else set()


def _bundle(program, patch):
    old, new = _parse(program).body, _parse(patch).body
    annotations = not _future_annotations(old + new)
    before = {ast.dump(node) for node in old}
    doubled = _double_entries(old)
    old_unkeyed = [node for node in old if _key(node) is None and not isinstance(node, _IMPORTS)]
    # Multiple writes to the same binding have an order-sensitive boundary; it matters only for names
    # this BUNDLE actually rebinds (a statement repeated unchanged rebinds nothing).
    repeated = _repeated(new)
    if repeated:
        raise _MergeError("ambiguous_binding", "the BUNDLE binds " + _names(repeated) + " more than once")
    touched = {name for node in new if ast.dump(node) not in before for name in _bindings(node)}
    repeated = _repeated(old) & touched
    if repeated:
        raise _MergeError("ambiguous_binding", "CURRENT PROGRAM binds " + _names(repeated)
                          + " more than once, so the BUNDLE cannot tell which binding it replaces")
    def first_entry():
        return next((i for i, previous in enumerate(old) if _has_call(previous)
                     or _key(previous) == ("entry_guard",)), len(old))
    for index, node in enumerate(new):
        key = _key(node)
        exact = [i for i, previous in enumerate(old) if ast.dump(previous) == ast.dump(node)]
        if exact:
            continue
        if key is None:
            if old_unkeyed:
                raise _MergeError("ambiguous_top_level", "CURRENT PROGRAM already has top-level statements that "
                                  "bind no name (" + _label(old_unkeyed[0]) + "), so the new " + _label(node)
                                  + " has no unambiguous place")
            old.append(node)
            continue
        names = set(_bindings(node))
        overlapping = [i for i, previous in enumerate(old) if names & set(_bindings(previous))]
        matching = [i for i, previous in enumerate(old) if _key(previous) == key]
        if len(matching) > 1 or names and overlapping != matching:
            raise _MergeError("ambiguous_binding", _label(node) + " does not match one statement of CURRENT "
                              "PROGRAM (" + "; ".join(_label(old[i]) for i in (overlapping or matching)[:2]) + ")")
        if matching:
            old[matching[0]] = node
        elif key == ("entry_guard",):
            old.append(node)
        elif key == ("module_doc",):
            old.insert(0, node)
        elif isinstance(node, _IMPORTS):
            pos = 1 if old and _key(old[0]) == ("module_doc",) else 0
            while pos < len(old) and (isinstance(old[pos], _IMPORTS) and
                    (not isinstance(node, ast.ImportFrom) or node.module != "__future__"
                     or isinstance(old[pos], ast.ImportFrom) and old[pos].module == "__future__")):
                pos += 1
            old.insert(pos, node)
        elif isinstance(node, _DEFS):
            # A new function or class goes before the first top-level call or guard, and before the next
            # statement of this BUNDLE that CURRENT PROGRAM already has (a helper written before its caller),
            # unless a name it reads while being defined is bound only after that statement: then the order
            # check below decides (a REPLACE is needed when the BUNDLE's own order cannot be kept).
            pos = first_entry()
            following = next((at for later in new[index + 1:] for at in (_existing(old, later),) if at is not None),
                             None)
            if following is not None and following < pos:
                need = _eager_names(node, annotations) & _bound(old + new)
                if need - _bound(old[:following]):
                    following = None
            old.insert(min(pos, following) if following is not None else pos, node)
        else:
            old.insert(first_entry(), node)
    twice = _double_entries(old) - doubled
    if twice:
        raise _MergeError("ambiguous_top_level", _names(twice) + " would run twice, from a bare top-level call "
                          "and from the __main__ guard; keep one entry")
    # Defaults/decorators can execute while definitions are being installed.
    # Do not silently reverse (or drop duplicates from) the submitted group; only an identical repeated
    # import is merged once.
    actual, cursor, previous, imports = [ast.dump(node) for node in old], 0, None, set()
    for node in new:
        if isinstance(node, _IMPORTS):
            if ast.dump(node) in imports:
                continue
            imports.add(ast.dump(node))
        try:
            cursor = actual.index(ast.dump(node), cursor) + 1
        except ValueError:
            raise _MergeError("bundle_order_requires_replace", _label(node) + " must come after "
                              + (_label(previous) if previous is not None else "the statements before it")
                              + " as in the BUNDLE, but it keeps its earlier place in CURRENT PROGRAM") from None
        previous = node
    source = ast.unparse(ast.Module(body=old, type_ignores=[]))
    return source.rstrip() + "\n" if source.strip() else ""


@dataclass(frozen=True)
class ApplyResult:
    accepted: bool
    changed: bool
    reason: str
    version: str


FORMAT_REASONS = ("no_code_block", "ambiguous_code_blocks", "split_code_blocks")
# The feedback of an accepted ready version that has not run yet.
READY = "Candidate ready for isolated execution; task success remains unverified."


def executed_note(version, result, *, layout=False):
    """The feedback of an accepted ready version that already ran (S2-08, thin's `loop` switch), in place of READY:
    that version is not executed again. `result` is the one-line result of its run (status, exit code, exception
    line); `layout`: this version is not that one byte for byte but the same code apart from comments and layout."""
    return ("Same as executed version " + version + (" apart from comments and layout" if layout else "") + ": "
            + result + ". It is not executed again; change the code to resolve that result.")


SYNTAX_NOTE = " SyntaxError at line "  # input_fidelity: starts the note on a syntax error inside kept code


def _inner_syntax(source, error):
    """input_fidelity: the feedback note on a syntax error inside the code of a length reply, not at its cut: a
    one-line string left open on a line before the last cannot come from the cut ('' otherwise)."""
    lines = source.rstrip().splitlines()
    detected = re.search(r"\(detected at line (\d+)\)", str(error.msg))
    if (not error.lineno or error.lineno >= len(lines)
            or not str(error.msg).startswith("unterminated string literal")
            or (detected and int(detected[1]) != error.lineno)):  # a string continued by a backslash may be the cut
        return ""
    text = lines[error.lineno - 1].strip()
    text = text if len(text) <= 160 else text[:157] + "..."
    return (SYNTAX_NOTE + f"{error.lineno} of the sent code: {str(error.msg)[:120]}: `{text}`. That is an error in "
            "the code, not the cut: the statement holding it and everything after it were dropped. Fix that line and "
            "resend the dropped work complete.")


def _syntax_detail(source):
    """'line N: message' of the code's first syntax error, or ''."""
    try:
        _parse(source)
    except SyntaxError as error:
        return (f"line {error.lineno}: " if error.lineno else "") + str(error.msg)[:120]
    except (ValueError, RecursionError, MemoryError, OverflowError):
        return ""
    return ""


class ProgramVersion:
    syntax_note = False  # input_fidelity (thin sets it): the feedback quotes a syntax error inside kept code

    def __init__(self, path="solution.py", expected_files=()):
        if not _valid_path(path):
            raise ValueError("path must be a single relative Python filename")
        self.path = path
        self.program = ""
        self.pending = ("entry",)
        self.advisories = ()
        self.feedback = ""
        self.expected_files = tuple(expected_files)  # deliverables that rank the code blocks of a reply
        self.extraction = None  # shape of the last admitted reply: raw, fenced or tolerant
        self._tails = set()
        self._dropped = ""  # first line of the dropped top-level statement behind incomplete:module
        # {version: one-line result of its run}: filled by thin's `loop` switch only (S2-08), so an accepted ready
        # version that already ran is told so (executed_note) instead of READY. Empty: READY as before.
        self.executed = {}

    @property
    def version(self):
        return hashlib.sha256(self.program.encode()).hexdigest()

    @property
    def ready(self):
        return bool(self.program) and not self.pending

    def apply(self, content, finish, mode, base_version, *, path=None):
        """Atomically admit a replacement or whole binding group against base SHA.

        A genuine cutoff can create a partial version containing only complete
        top-level statements. No tail text is stored or joined to future replies.
        Complete replacement is also how a caller abandons old unfinished work.
        Rejection changes feedback only; program, version and obligations survive.
        """
        def reject(reason, detail=""):
            action = ("use REPLACE with a complete file against the current version"
                      if reason in ("ambiguous_binding", "ambiguous_top_level", "bundle_order_requires_replace")
                      else "send all the code in ONE ```python block after the header line; blocks are never joined"
                      if reason in FORMAT_REASONS else "resubmit a complete group against the current version")
            self.feedback = reason + (": " + detail if detail else "") + "; " + action
            return ApplyResult(False, False, reason, self.version)

        if base_version != self.version:
            # Only a model-written header can name another version: restate the one required now, inline.
            return reject("stale_base", "the header named another version; the first line must be REPLACE or "
                          "BUNDLE followed by " + self.version)
        if path is not None and (not _valid_path(path) or path != self.path):
            return reject("invalid_path")
        if mode not in ("replace", "bundle"):
            return reject("invalid_mode")
        if not isinstance(content, str):
            return reject("invalid_format")
        content = _closing_fence_cut(content, finish)
        tail = ""
        inner = ""
        # A BUNDLE against an empty program is the whole file: merged and extracted as a REPLACE (the
        # obligations still follow the BUNDLE rule: unfinished work it does not complete stays pending).
        merge = "replace" if mode == "bundle" and not self.program else mode
        try:
            extraction = extract_code(content, finish, mode=merge, expected_files=self.expected_files,
                                      program=self.program)
            source = extraction.source
            if not source:
                return reject("empty_code", "the code block is empty")
            try:
                incoming = _parse(source)
            except SyntaxError as error:
                # Only code that ends where a length reply was cut may keep a verified prefix; an error on
                # that last line of a block still open at the cut is the cut whatever its shape (`x = np.`,
                # `a if b els`). In a closed block it is the model's own error.
                at_cut = _still_open(content) and error.lineno == len(source.rstrip().splitlines())
                if not extraction.cut or not (at_cut or _cutoff(source, error)):
                    return reject("invalid_syntax", _syntax_detail(source))
                if self.syntax_note:
                    inner = _inner_syntax(source, error)
                source, tail = continuation.verified_prefix(source, True)
                incoming = _parse(source)
            else:
                if extraction.cut:
                    # Code that still compiles at a cut: its unfinished last statement is not kept.
                    source, tail = _open_cut(content, source)
                    incoming = _parse(source) if source.strip() else ast.Module(body=[], type_ignores=[])
            candidate = source if merge == "replace" else _bundle(self.program, source)
            tree = _parse(candidate)
        except FormatError as error:
            return reject(error.reason, error.detail)
        except _MergeError as error:
            return reject(error.reason, error.detail)
        except ValueError as error:
            return reject(str(error) if str(error) in ("invalid_format", "ambiguous_binding",
                                                       "ambiguous_top_level", "bundle_order_requires_replace")
                          else "invalid_syntax")
        except (SyntaxError, RecursionError, MemoryError, OverflowError):
            return reject("invalid_syntax")

        tail = _code_lines(tail)  # a cut comment owes nothing
        defined = {node.name for node in tree.body if isinstance(node, _DEFS)}
        before = {ast.dump(node) for node in (_parse(self.program).body if self.program else [])}
        tails = set() if mode == "replace" else set(self._tails)
        dropped = "" if mode == "replace" else self._dropped
        completed = {name for node in incoming.body for name in _bindings(node)
                     if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     or not _placeholder(node)}
        tails.difference_update(completed)
        # Dropped top-level work that no binding names (incomplete:module) and a dropped __main__ guard
        # (entry) are resupplied by new top-level work of an accepted reply (_resupplies): a statement it
        # repeats unchanged from CURRENT PROGRAM, a bare library call, a binding or a definition never
        # clears them, and a reply cut in top-level work again records its own.
        if tails & {None, _GUARD} and any(_resupplies(node, defined) for node in incoming.body
                                          if ast.dump(node) not in before):
            tails -= {None, _GUARD}
        if tail.strip():
            owed = _tail_name(tail)
            tails.add(owed)
            if owed is None:
                dropped = tail.split("\n", 1)[0].strip()[:80]
        # A complete __main__ guard that calls something is the program's entry: top-level work cut outside
        # it is reported once (accepted_partial) but not owed.
        if _runs_guard(tree.body):
            tails -= {None, _GUARD}
        dropped = dropped if None in tails else ""
        pending = ["incomplete:" + (name or "module")
                   for name in sorted(tails - {_GUARD}, key=lambda value: value or "")]
        pending.extend("stub:" + node.name for node in ast.walk(tree)
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and _placeholder(node))
        if any(_noop(node) for node in tree.body):
            pending.append("stub:module")
        if _GUARD in tails or not _has_entry(tree.body, defined):
            pending.append("entry")
        changed = candidate != self.program
        self.program, self._tails, self.extraction = candidate, tails, extraction.shape
        self._dropped = dropped
        self.pending = tuple(dict.fromkeys(pending))
        self.advisories = tuple("unresolved_call:" + name for name in continuation.missing_calls(candidate))
        self.feedback = ("Pending: " + ", ".join(self.pending) + ". Resend complete groups, not literal fragments."
                         if self.pending else executed_note(self.version, self.executed[self.version])
                         if self.version in self.executed else READY)
        if dropped:
            self.feedback += (" incomplete:module: the top-level statement starting `" + dropped + "` was cut and "
                              "dropped with everything after it; put that work in functions called from one "
                              'complete `if __name__ == "__main__":` guard.')
        if _GUARD in tails:
            self.feedback += (' entry: the `if __name__ == "__main__":` guard was cut and dropped; send the complete '
                              "guard again (a BUNDLE with the guard is enough when the functions it calls exist).")
        if inner:
            self.feedback += inner
        if self.advisories:
            self.feedback += " Advisory: " + ", ".join(self.advisories)
        return ApplyResult(True, changed, "accepted_partial" if tail.strip() else "accepted", self.version)
