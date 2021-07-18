import h5py
import numpy as np

readpath = '/users/jsolt/data/shared/LaPlanteSims/v10/t21_snapshots_wedge.hdf5'
writepath = '/users/jsolt/data/shared/LaPlanteSims/v10/t21_snapshots_wedge_fft.hdf5'
groups = ['ksz_snapshots', 'layer_redshifts', 'snapshot_labels', 't21_snapshots']

d = {}
print("Loading data...")
snapshots = None
with h5py.File(readpath, 'r') as f:
    d['snapshot_labels'] = f['Data']['snapshot_labels'][:500]
    snapshots = f['Data']['t21_snapshots'][:500]

print("Fourier transforming...")
snapshots = np.fft.fft2(snapshots)
d['t21_snapshots_fft_real'] = snapshots.real.astype(np.dtype('<f4'))
d['t21_snapshots_fft_imag'] = snapshots.imag.astype(np.dtype('<f4'))
print(d['t21_snapshots_fft_real'].shape)
print(d['t21_snapshots_fft_imag'].shape)

print("Saving...")
with h5py.File(writepath, 'w') as f:
    f.create_group('Data')
    f['Data'].create_dataset("snapshot_labels", data=d['snapshot_labels'], chunks=True, maxshape=(None, 6))
    f['Data'].create_dataset("t21_snapshots_fft_real", data=d['t21_snapshots_fft_real'], chunks=True, maxshape=(None, 30, 512, 512))
    f['Data'].create_dataset("t21_snapshots_fft_imag", data=d['t21_snapshots_fft_imag'], chunks=True, maxshape=(None, 30, 512, 512))
    
    print(list(f['Data'].keys()))
    print(f['Data']['t21_snapshots_fft_real'])

d = {}
print("Loading data...")
snapshots = None
with h5py.File(readpath, 'r') as f:
    d['snapshot_labels'] = f['Data']['snapshot_labels'][500:]
    snapshots = f['Data']['t21_snapshots'][500:]

print("Fourier transforming...")
snapshots = np.fft.fft2(snapshots)
d['t21_snapshots_fft_real'] = snapshots.real.astype(np.dtype('<f4'))
d['t21_snapshots_fft_imag'] = snapshots.imag.astype(np.dtype('<f4'))
print(d['t21_snapshots_fft_real'].shape)
print(d['t21_snapshots_fft_imag'].shape)

print("Saving...")
with h5py.File(writepath, 'a') as f:
    #f.create_group('Data')
    f['Data']['snapshot_labels'].resize((f['Data']['snapshot_labels'].shape[0] + d['snapshot_labels'].shape[0]), axis = 0)
    f['Data']['snapshot_labels'][-d['snapshot_labels'].shape[0]:] = d['snapshot_labels']
    
    f['Data']['t21_snapshots_fft_real'].resize((f['Data']['t21_snapshots_fft_real'].shape[0] + d['t21_snapshots_fft_real'].shape[0]), axis = 0)
    f['Data']['t21_snapshots_fft_real'][-d['t21_snapshots_fft_real'].shape[0]:] = d['t21_snapshots_fft_real']
    
    f['Data']['t21_snapshots_fft_imag'].resize((f['Data']['t21_snapshots_fft_imag'].shape[0] + d['t21_snapshots_fft_imag'].shape[0]), axis = 0)
    f['Data']['t21_snapshots_fft_imag'][-d['t21_snapshots_fft_imag'].shape[0]:] = d['t21_snapshots_fft_imag']
    
    print(list(f['Data'].keys()))
    print(f['Data']['t21_snapshots_fft_real'])
