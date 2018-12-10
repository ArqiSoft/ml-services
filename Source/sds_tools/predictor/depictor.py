from rdkit import Chem
from rdkit.Chem import AllChem
from IPython.display import SVG
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


fptype = {'Type': 'FCFC','Size':256,'Radius':2} #type of used Morgan FP
mols = ['<RDKit mol object>']
bitId = str(4) #string number of bit in Morgan vector
zid = 0 #int number of molecule in dataframe if importing several mols


def _prepareMol(mol,kekulize):
    """
    kekulize mol
    :param mol:
    :param kekulize:
    :return:
    """
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    return mc


def moltosvg(mol,molSize=(450,200),kekulize=True,drawer=None,**kwargs):
    """
    optimized for jupiter notebooks, need correction for .py
    :param mol:
    :param molSize:
    :param kekulize:
    :param drawer:
    :param kwargs:
    :return:
    """
    mc = _prepareMol(mol,kekulize)
    if drawer is None:
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mc,**kwargs)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return SVG(svg.replace('svg:',''))


def getSubstructDepiction(mol,atomID,radius,molSize=(450,200)):
    """
    do a depiction where the atom environment is highlighted normally and the central atom
    is highlighted in blue

    :param mol:
    :param atomID:
    :param radius:
    :param molSize:
    :return:
    """
    if radius>0:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol,radius,atomID)
        atomsToUse=[]
        for b in env:
            atomsToUse.append(mol.GetBondWithIdx(b).GetBeginAtomIdx())
            atomsToUse.append(mol.GetBondWithIdx(b).GetEndAtomIdx())
        atomsToUse = list(set(atomsToUse))
    else:
        atomsToUse = [atomID]
        env=None
    return moltosvg(mol,molSize=molSize,highlightAtoms=atomsToUse,highlightAtomColors={atomID:(0.3,0.3,1)})


def depictBit(zid,bitId,mols,fptype={'Type': 'FCFC','Size':256,'Radius':2},molSize=(225,100)):
    """

    :param zid:
    :param bitId:
    :param mols:
    :param fptype:
    :param molSize:
    :return:
    """
    info = {}
    if fptype['Type'] == 'FCFC' or fptype['Type'] == 'ECFC':
        fp = AllChem.GetHashedMorganFingerprint(mols[zid],fptype['Radius'],fptype['Size'],useFeatures=True,bitInfo=info)
    elif fptype['Type'] == 'ECFP' or fptype['Type'] == 'FCFP':
        fp = AllChem.GetHashedMorganFingerprint(mols[zid],fptype['Radius'],fptype['Size'],useFeatures=False,bitInfo=info)
    aid,rad = info[int(bitId)][0]
    return getSubstructDepiction(mols[zid],aid,rad,molSize=molSize)


depictBit(zid=zid,bitId=bitId,mols=mols,fptype=fptype,molSize=(300,150))
