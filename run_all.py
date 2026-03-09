import subprocess, sys, time, argparse

STEPS = [
    (1,  'notebooks/01_data_loading.py',         'Data Loading'),
    (2,  'notebooks/02_eda.py',                   'EDA'),
    (3,  'notebooks/03_missing_data.py',          'Missing Data'),
    (4,  'notebooks/04_feature_engineering.py',   'Feature Engineering'),
    (5,  'notebooks/05_feature_selection.py',     'Feature Selection'),
    (6,  'notebooks/06_imbalance.py',              'Imbalance Strategy'),
    (7,  'notebooks/07_baseline_models.py',        'Baseline Models'),
    (8,  'notebooks/08_tree_models.py',            'XGBoost'),
    (9,  'notebooks/09_threshold_optimization.py', 'Threshold Optimization'),
    (10, 'notebooks/10_evaluation.py',             'Final Evaluation'),
]

def run(num, path, name):
    print('STEP ' + str(num) + ': ' + name)
    t = time.time()
    r = subprocess.run([sys.executable, path])
    ok = r.returncode == 0
    print('Done in ' + str(int(time.time()-t)) + 's - ' + ('OK' if ok else 'FAILED'))
    return ok

parser = argparse.ArgumentParser()
parser.add_argument('--from', dest='from_step', type=int, default=1)
parser.add_argument('--only', dest='only_step', type=int, default=None)
args = parser.parse_args()

steps = [s for s in STEPS if s[0] == args.only_step] if args.only_step else [s for s in STEPS if s[0] >= args.from_step]

print('PayGuard - Running ' + str(len(steps)) + ' steps')
for num, path, name in steps:
    if not run(num, path, name):
        print('FAILED at step ' + str(num))
        break
