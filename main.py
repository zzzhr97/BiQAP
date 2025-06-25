from src.utils.parser import get_args
from src.runner import Trainer, Tester

def run(args, cfg_file):
    if args.mode == 'train':
        runner = Trainer(cfg_file)
    elif args.mode == 'test':
        runner = Tester(cfg_file)
        
    runner.run()
    runner.end()
    return runner.id() 

def main():
    args = get_args('Deep learning of QAP model trianing & evaluation code.')

    ids = []
    for cfg_f in args.cfg_files:
        id = run(args, cfg_f)
        ids.append(id)
        
    print(f'All done:')
    for id in ids:
        print(f'\t{id}')
        
if __name__ == '__main__':

    main()