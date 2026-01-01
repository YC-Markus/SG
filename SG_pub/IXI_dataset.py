import h5py
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Tuple, Dict

class IXIDataset(Dataset):
    def __init__(self,
                 h5_paths: List[str],
                 slice_percentage: Optional[float] = 0.85,
                 ):
        
        self.h5_paths = h5_paths
        self.slice_percentage = slice_percentage

        self.case_to_path: Dict[str, str] = {}  # case_id -> file_path
        self.index_map: List[Tuple[str, int]] = []  # (case_id, slice_idx)
        self.h5_file_handles: Dict[str, h5py.File] = {}  # path -> h5 file

        for path in self.h5_paths:
            with h5py.File(path, 'r') as f:
                for cid in sorted(list(f.keys())):
                    if cid in self.case_to_path:
                        raise ValueError(f"Duplicate case ID '{cid}' found in multiple files.")
                    self.case_to_path[cid] = path

                    d = f[cid]['PD'].shape[2]
                    
                    if self.slice_percentage is None:
                        selected = range(d)
                    else:
                        if not (0.0 <= self.slice_percentage <= 1.0):
                             raise ValueError("slice_percentage must be from 0.0 to 1.0")
                        
                        n = int(round(d * self.slice_percentage))
                        if d > 0 and n == 0 and self.slice_percentage > 0:
                            n = 1
                        
                        n = min(n, d) 
                        selected = range(n)
                    self.index_map.extend([(cid, s) for s in selected])

    def __len__(self):
        return len(self.index_map)

    def _get_file(self, path: str):
        if path not in self.h5_file_handles:
            self.h5_file_handles[path] = h5py.File(path, 'r')
        return self.h5_file_handles[path]

    def __getitem__(self, idx: int):
        case_id, slice_pos = self.index_map[idx]
        file_path = self.case_to_path[case_id]
        h5_file = self._get_file(file_path)
        g = h5_file[case_id]

        pd_slice = g['PD'][:, :, slice_pos]
        t2_slice = g['T2'][:, :, slice_pos]
        t1_slice = g['T1'][:, :, slice_pos]

        pd_slice = (pd_slice - 0) / (g['PD'].attrs['p99_5'] - 0)
        t2_slice = (t2_slice - 0) / (g['T2'].attrs['p99_5'] - 0)
        t1_slice = (t1_slice - 0) / (g['T1'].attrs['p99_5'] - 0)
        to_tensor = lambda x: torch.clamp(torch.tensor(x, dtype=torch.float32).unsqueeze(0), 0, 1)

        return {
            'PD': to_tensor(pd_slice),
            'T2': to_tensor(t2_slice),
            'T1': to_tensor(t1_slice),
            'case_id': case_id,
            'slice_idx': slice_pos,
        }

    def __del__(self):
        for f in self.h5_file_handles.values():
            try:
                f.close()
            except:
                pass