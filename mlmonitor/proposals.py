"""Agent interface for proposals and outcomes. User review happens in the UI."""
import argparse
import json
from pathlib import Path
from urllib.request import urlopen

from .advice import ProposalStore, opportunities
from .config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8790/api/metrics')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list')
    commands.add_parser('suggest')
    submit = commands.add_parser('submit')
    submit.add_argument('--run', required=True)
    submit.add_argument('--plan', type=Path, required=True)
    submit.add_argument('--opportunity')
    outcome = commands.add_parser('outcome')
    outcome.add_argument('--id', required=True)
    outcome.add_argument('--content-sha256', required=True)
    outcome.add_argument('--status', choices=['completed', 'failed', 'reverted'], required=True)
    outcome.add_argument('--summary-file', type=Path, required=True)
    args = parser.parse_args()
    settings = load_config(args.config)['decision_support']
    if not settings['enabled']:
        parser.error('decision support is disabled; the user must opt in first')
    store = ProposalStore(settings['directory'])
    if args.command == 'outcome':
        result = store.record_outcome(args.id, args.content_sha256, args.status, args.summary_file.read_text())
    else:
        with urlopen(args.url, timeout=10) as response:
            payload = response.read(32_000_001)
        if len(payload) > 32_000_000:
            parser.error('snapshot exceeds 32 MB')
        snapshot = json.loads(payload)
        if snapshot.get('loading'):
            parser.error('monitor has not collected a snapshot yet')
        if args.command == 'submit':
            if args.plan.stat().st_size > 100_000:
                parser.error('plan exceeds 100 KB')
            result = store.submit(snapshot, args.run, json.loads(args.plan.read_text()), args.opportunity)
        elif args.command == 'suggest':
            result = opportunities(snapshot)
        else:
            result = store.listing(snapshot)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
