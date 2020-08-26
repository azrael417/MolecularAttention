import io

import cairosvg
import numpy as np
from PIL import Image
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from features.utils import Invert

# now you can import sascore!

smiles_vocab = None  # load charater to int function
data_location = 'data/'

MORDRED_SIZE = 1613


def smile_to_mordred(smi, imputer_dict=None, userdkit=False):
    calc = Calculator(descriptors, ignore_3D=True)
    if userdkit:
        smi = Chem.MolFromSmiles(smi)
        assert (smi is not None)
    res = calc(smi)
    res = np.array(list(res.values())).reshape(1, -1).astype(np.float32)
    res = np.nan_to_num(res, posinf=0, neginf=0, nan=0)
    if imputer_dict is not None:
        imputer_dict = imputer_dict[0]
        res = imputer_dict['scaler'].transform(imputer_dict['imputer'].transform(res))
    return res.flatten().astype(np.float32)


def smiles_to_image(mol, molSize=(128, 128), kekulize=True, mol_name='', mol_computed=True):
    if not mol_computed:
        mol = Chem.MolFromSmiles(mol)
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    print('mol', mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    print(svg)
    ios = cairosvg.svg2png(bytestring=svg, parent_width=100, parent_height=100,scale=1)
    ios = io.BytesIO()
    image = Image.open(ios)
    image.convert('RGB')
    return (Invert()(image))
