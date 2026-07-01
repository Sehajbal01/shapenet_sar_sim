import torch
import itertools
from utils import cartesian_to_spherical, spherical_to_cartesian

def sign_str(v):
    return f'+{int(v)}' if v > 0 else f'{int(v)}'

print('cartesian_to_spherical for all ±1 combinations:')
for x, y, z in itertools.product([1, -1], repeat=3):
    cart = torch.tensor([float(x), float(y), float(z)])
    az, el, dist = cartesian_to_spherical(cart)
    print(f'  x,y,z = {sign_str(x)},{sign_str(y)},{sign_str(z)}  ->  az={az.item():+7.2f} deg,  el={el.item():+7.2f} deg,  dist={dist.item():.4f}')

print()
print('round-trip check (cartesian -> spherical -> cartesian):')
all_pass = True
for x, y, z in itertools.product([1, -1], repeat=3):
    cart_in = torch.tensor([float(x), float(y), float(z)])
    az, el, dist = cartesian_to_spherical(cart_in)
    cart_out = spherical_to_cartesian(az, el, dist)
    err = (cart_in - cart_out).abs().max().item()
    status = 'PASS' if err < 1e-5 else 'FAIL'
    if status == 'FAIL':
        all_pass = False
    print(f'  x,y,z = {sign_str(x)},{sign_str(y)},{sign_str(z)}  ->  err={err:.2e}  {status}')

print()
print('All round-trip tests passed!' if all_pass else 'SOME TESTS FAILED!')
