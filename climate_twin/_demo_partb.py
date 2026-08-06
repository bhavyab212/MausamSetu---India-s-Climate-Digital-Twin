import warnings, os
warnings.filterwarnings('ignore'); os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np, torch
import data_source as DS
from training.loops import RoundConfig, TrainingProgress, train_one_round
from training.model import ClimateTwinModel
from training import registry as REG

REG.init_registry(); region = 'india'
rain, temp, mask, years = DS.load_aggregates(region); _lm = mask == 1
rn = np.clip((rain - np.nanmin(rain[:, _lm])) / (np.nanmax(rain[:, _lm]) - np.nanmin(rain[:, _lm]) + 1e-8), 0, 1)
tn = np.clip((temp - np.nanmin(temp[:, _lm])) / (np.nanmax(temp[:, _lm]) - np.nanmin(temp[:, _lm]) + 1e-8), 0, 1)
full = np.stack([rn, tn], -1); seq = 5
arch = {'seq_length': seq, 'hidden': 32, 'channels': 2, 'residual_scale': 0.1, 'grid': list(mask.shape)}

def windows(sl):
    X = np.array([sl[i:i + seq] for i in range(len(sl) - seq)])
    Y = np.array([sl[i + seq] for i in range(len(sl) - seq)])
    return X, Y

def mk():
    return ClimateTwinModel(seq_length=seq, lat_dim=mask.shape[0], lon_dim=mask.shape[1],
                            channels=2, hidden=32, dropout=0.1)

tr = full[0:12]; X, Y = windows(tr); vs = full[7:13]; Xv, Yv = windows(vs)

# V1 fresh
print('Training V1 (fresh, 4 epochs)...')
cfg = RoundConfig(); cfg.seq_length = seq; cfg.epochs = 4; cfg.batch_size = 8; cfg.seed = 42
m, met1 = train_one_round(X, Y, Xv, Yv, mask, cfg, TrainingProgress(), region=region)
REG.save_model('demo_v1', region, m, arch, met1, epochs_add=4, rounds_add=1, notes='new')
print('V1  rmse=%.5f' % met1['rmse'])

# V2 continue from v1
print('\nTraining V2 (continue from v1, +4 epochs)...')
m2 = mk(); ok, _ = REG.load_into('demo_v1', region, m2); assert ok, 'resume failed'
cfg2 = RoundConfig(); cfg2.seq_length = seq; cfg2.epochs = 4; cfg2.batch_size = 8; cfg2.seed = 42
m2, met2 = train_one_round(X, Y, Xv, Yv, mask, cfg2, TrainingProgress(), model=m2, region=region)
REG.save_model('demo_v1_v2', region, m2, arch, met2, epochs_add=4, rounds_add=1,
               notes='resumed continue', parent_name='demo_v1')
print('V2  rmse=%.5f  (continued from v1, warm-start OK)' % met2['rmse'])

# FT fine-tune v2: low LR + freeze core + recent slice
print('\nTraining FT (fine-tune v2, LR/10, frozen core, recent-5yr)...')
m3 = mk(); REG.load_into('demo_v1_v2', region, m3)
sl = full[len(full) - 10:]; Xr, Yr = windows(sl)
cfg3 = RoundConfig(); cfg3.seq_length = seq; cfg3.epochs = 4; cfg3.batch_size = 8
cfg3.lr = 1e-4; cfg3.freeze_recurrent = True; cfg3.warmup_epochs = 0; cfg3.seed = 42
m3, met3 = train_one_round(Xr, Yr, Xr[-2:], Yr[-2:], mask, cfg3, TrainingProgress(), model=m3, region=region)
REG.save_model('demo_v1_v2_ft_recent', region, m3, arch, met3, epochs_add=4, rounds_add=1,
               notes='resumed fine-tune', parent_name='demo_v1_v2')
print('FT  rmse=%.5f  (LR=1e-4, frozen core, recent-5yr slice)' % met3['rmse'])

print('\n--- LINEAGE TREE ---')
def show(n, d=0):
    print('   ' * d + ('└─ ' if d else '📦 ') + n['name'] + (('  ← ' + n['parent']) if n['parent'] else '  (root)'))
    for c in n['children']:
        show(c, d + 1)
for t in REG.get_lineage_tree(region):
    if 'demo' in t['name']:
        show(t)

print('\n--- COMPARISON (winner = lowest RMSE) ---')
rows = [(n, REG.get_model(n, region)['metrics']) for n in ['demo_v1', 'demo_v1_v2', 'demo_v1_v2_ft_recent']]
best = min(rows, key=lambda r: r[1].get('rmse', 9e9))[0]
for n, mm in rows:
    tag = '  ← WINNER' if n == best else ''
    print('  %-28s rmse=%.5f  csi=%.4f%s' % (n, mm.get('rmse', float('nan')), mm.get('csi', float('nan')), tag))

print('\nCleaning up demo models...')
for n in ['demo_v1', 'demo_v1_v2', 'demo_v1_v2_ft_recent']:
    REG.delete_model(n, region)
print('Done.')
