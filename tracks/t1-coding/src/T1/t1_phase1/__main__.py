import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description='Track 1 phase 1 local research harness')
    commands=parser.add_subparsers(dest='command',required=True)
    solve=commands.add_parser('solve')
    solve.add_argument('--task-dir',type=Path,required=True)
    solve.add_argument('--out',type=Path,required=True)
    prepare=commands.add_parser('prepare')
    prepare.add_argument('--participant',type=Path,required=True)
    prepare.add_argument('--root',type=Path,default=Path.cwd())
    preflight=commands.add_parser('preflight')
    evaluate=commands.add_parser('evaluate')
    for command in (preflight,evaluate):
        command.add_argument('--context',default='colima')
        command.add_argument('--image',default='t1-phase1:b0-amd64')
        command.add_argument('--platform',default='linux/amd64')
    evaluate.add_argument('--root',type=Path,default=Path.cwd())
    evaluate.add_argument('--run-dir',type=Path,required=True)
    evaluate.add_argument('--split',choices=('dev','holdout','all'),default='dev')
    evaluate.add_argument('--unit-timeout',type=int,default=420)
    evaluate.add_argument('--total-budget',type=int,default=7200)
    evaluate.add_argument('--checker-timeout',type=int,default=600)
    evaluate.add_argument('--integration-evidence',type=Path)
    args=parser.parse_args()
    if args.command=='solve':
        from .baseline import solve, solve_budget
        budget,source=solve_budget(args.task_dir)
        value=solve(args.task_dir,args.out,budget_sec=budget,budget_source=source)
    elif args.command=='prepare':
        from .evaluator import prepare
        result=prepare(args.root.resolve(),args.participant.resolve())
        value={'staged_units':len(result['units']),'source_commit':result['source_commit']}
    elif args.command=='preflight':
        from .evaluator import preflight
        value=preflight(args.context,args.image,args.platform)
    else:
        from .evaluator import evaluate
        result=evaluate(args.root.resolve(),args.run_dir.resolve(),split=args.split,context=args.context,
            image=args.image,platform=args.platform,unit_timeout=args.unit_timeout,
            total_budget=args.total_budget,checker_timeout=args.checker_timeout,
            integration_evidence=args.integration_evidence)
        value=result['summary']
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
