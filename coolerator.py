#!/usr/bin/env python

from __future__ import division
import sys
import numpy as np
from numpy import sin, cos
from molecule import *
import itertools
from collections import defaultdict, OrderedDict
import pickle
import networkx

icetype = raw_input("Enter your favorite ice (III, V, VI, VII) -> ")

def format_box(mat):
    return "[%.3f %.3f %.3f], [%.3f %.3f %.3f], [%.3f %.3f %.3f]" % (mat[0][0],mat[0][1],mat[0][2],
                                                                     mat[1][0],mat[1][1],mat[1][2],
                                                                     mat[2][0],mat[2][1],mat[2][2])

def GetLatticeVectors(a, b, c, alpha, beta, gamma):
    alpha *= np.pi/180
    beta  *= np.pi/180
    gamma *= np.pi/180
    v = np.sqrt(1 - cos(alpha)**2 - cos(beta)**2 - cos(gamma)**2 + 2*cos(alpha)*cos(beta)*cos(gamma))
    Mat = np.mat([[a, b*cos(gamma), c*cos(beta)],
                  [0, b*sin(gamma), c*((cos(alpha)-cos(beta)*cos(gamma))/sin(gamma))],
                  [0, 0, c*v/sin(gamma)]])
    L1 = Mat*np.mat([[1],[0],[0]])
    L2 = Mat*np.mat([[0],[1],[0]])
    L3 = Mat*np.mat([[0],[0],[1]])
    return L1, L2, L3

OccDict = OrderedDict()
if icetype.lower() == 'iii':
    a = 6.676
    b = 6.676
    c = 6.955
    alpha = 90.0
    beta =  90.0
    gamma = 90.0
    ma = 4
    mb = 4
    mc = 3
    fnm = 'Raw/IceIII.xyz'
    box = GetLatticeVectors(a*ma,b*mb,c*mc,alpha,beta,gamma)
    OccDict["ha"] = 1/3
    OccDict["hb"] = 1/2
    OccDict["hc"] = 2/3
elif icetype.lower() == 'v':
    # Original monoclinic cell
    # a = 9.09
    # b = 7.55
    # c = 10.25
    # alpha = 90.0
    # beta = 109.1
    # gamma = 90.0
    # ma = 3
    # mb = 3
    # mc = 2
    # fnm = 'Raw/IceV.xyz'
    # box = GetLatticeVectors(a*ma,b*mb,c*mc,alpha,beta,gamma)
    # New orthogonal cell
    
    a = 17.722
    b = 22.650
    c = 34.957
    alpha = 90.0
    beta =  90.0
    gamma = 90.0
    ma = 2
    mb = 1
    mc = 1
    fnm = 'Raw/IceV_Orth2.xyz'
    box = GetLatticeVectors(a*ma,b*mb,c*mc,alpha,beta,gamma)
    
    OccDict["he"] = 0.56
    OccDict["hf"] = 0.44
    OccDict["hg"] = 0.44
    OccDict["hh"] = 0.29
    OccDict["hi"] = 0.51
    OccDict["hj"] = 0.76
    OccDict["hk"] = 0.56
    OccDict["hl"] = 0.24
    OccDict["hm"] = 0.49
    OccDict["hn"] = 0.71
    OccDict["ho"] = 0.49
    OccDict["hp"] = 0.50
    OccDict["hq"] = 0.51
    OccDict["hr"] = 0.50
elif icetype.lower() == 'vii':
    fnm = 'Raw/IceVII.xyz'
    box=[np.array([26.8,0,0]),np.array([0,26.8,0]),np.array([0,0,26.8])]
elif icetype.lower() == 'vi':
    fnm = 'Raw/IceVI_444.xyz'
    box=[np.array([24.724,0,0]),np.array([0,24.724,0]),np.array([0,0,22.792])]
else:
    raise Exception("This ice isn't supported yet")

M = Molecule(fnm)

class HBond:
    def __init__(self, oa, ob, ha, hb, occa=0.5, occb=0.5):
        # The two oxygens that are involved in the hydrogen bond.
        self.OA = oa
        self.OB = ob
        # The hydrogen that is closer to oxygen A.
        self.HA = ha
        self.HB = hb
        self.occa = occa
        self.occb = occb
        # The oxygen that CURRENTLY possesses the hydrogen.
        self.O = None
    def randomize(self):
        # Randomly assign the hydrogen to one of the oxygens.
        rnd = np.random.random()
        if rnd < self.occa:
            self.O = self.OA
        else:
            self.O = self.OB
    def displace(self,coord=None,occn=None,elem=None):
        # Occupancy shift.
        if self.O == self.OA:
            self.O = self.OB
            if coord != None:
                coord[self.OB] += 1
                coord[self.OA] -= 1
            if occn != None and elem != None:
                occn[elem[self.HB]] += 1
                occn[elem[self.HA]] -= 1
        elif self.O == self.OB:
            self.O = self.OA
            if coord != None:
                coord[self.OA] += 1
                coord[self.OB] -= 1
            if occn != None and elem != None:
                occn[elem[self.HA]] += 1
                occn[elem[self.HB]] -= 1

class System:
    def __init__(self, HBonds, Oxygens, Elem):
        self.HBonds = HBonds
        for hb in self.HBonds:
            hb.randomize()
        self.Oxygens = Oxygens
        self.Coord = defaultdict(int)
        for hb in self.HBonds:
            self.Coord[hb.O] += 1
        # Occupations go by hydrogen instead of oxygen.
        self.OccN = defaultdict(int)
        self.OccD = defaultdict(int)
        self.Elem = [i.lower() for i in Elem]
        for hb in self.HBonds:
            if hb.O == hb.OA:
                self.OccN[self.Elem[hb.HA]] += 1
            else:
                self.OccN[self.Elem[hb.HB]] += 1
            self.OccD[self.Elem[hb.HA]] += 1
            self.OccD[self.Elem[hb.HB]] += 1
        
    def Violation(self):
        return np.sum([np.abs(self.Coord[o]-2) for o in self.Coord])
    #return sum([sum([HB.O == ox for HB in self.HBonds]) != 2 for ox in self.Ox])
    def Anneal(self):
        viol = 1
        cyc = 0
        w = 100
        while viol > 0:
            cyc += 1
            # Pick a hydrogen bond at random.
            hb = self.HBonds[np.random.randint(len(self.HBonds))]
            Bias = True
            if Bias:
                if hb.O == hb.OA:
                    H1 = hb.HA
                    H2 = hb.HB
                    O1 = hb.occa
                    O2 = hb.occb
                else:
                    H1 = hb.HB
                    H2 = hb.HA
                    O1 = hb.occb
                    O2 = hb.occa
                sreal = self.OccN[self.Elem[H1]]/self.OccD[self.Elem[H1]]
                srealp = self.OccN[self.Elem[H2]]/self.OccD[self.Elem[H2]]
                if sreal + srealp != 1.0:
                    raise
                dsreal = srealp - sreal
                dsideal = O2 - O1
                crit = min(1,np.exp(-w*(dsreal-dsideal)))
                if np.random.random() > crit: 
                    continue
            cd0 = np.abs(self.Coord[hb.OA] - self.Coord[hb.OB])
            if cd0 == 0: continue
            hb.displace(self.Coord, self.OccN, self.Elem)
            cd1 = np.abs(self.Coord[hb.OA] - self.Coord[hb.OB])
            if cd1 > cd0:
                hb.displace(self.Coord, self.OccN, self.Elem)
            elif cd1 == cd0:
                if np.random.random() < 0.5:
                    hb.displace(self.Coord, self.OccN, self.Elem) # Reject the move with 50% probability.
            lastviol = viol
            viol = self.Violation()
            if viol < lastviol:
                print "\rAfter %i cycles, the sum of absolute coordination violations is now %i  " % (cyc,viol),
    def Select(self):
        AtomSelect = []
        for hb in self.HBonds:
            AtomSelect.append(hb.O)
            if hb.O == hb.OA:
                AtomSelect.append(hb.HA)
            else:
                AtomSelect.append(hb.HB)
        return np.array([int(i) for i in set(AtomSelect)])
    # def Violated(self):
        
        # print [sum([HB.O == ox for HB in self.HBonds]) != 2]

AllHBonds = []

shifts = []
for i in [0,1,-1]:
    for j in [0,1,-1]:
        for k in [0,1,-1]:
            shifts.append(i*box[0]+j*box[1]+k*box[2])

def pbcdx(xi, xj, cutoff=3.0):
    # This might not work for really tiny boxes.
    ndx = cutoff
    for s in shifts:
        dx = xj[0]-xi[0]-s[0]
        if np.abs(dx) > cutoff:
            continue
        dy = xj[1]-xi[1]-s[1]
        if np.abs(dy) > cutoff:
            continue
        dz = xj[2]-xi[2]-s[2]
        if np.abs(dz) > cutoff:
            continue
        ndx = np.sqrt(dx**2+dy**2+dz**2)
        break
    return ndx

Os = [j for j, b in enumerate(M.elem) if b[0] == 'O']

HOFile = os.path.splitext(fnm)[0]+".hof"
if os.path.exists(HOFile):
    print "Loading hydrogen-oxygen contacts from file.."
    HODict = pickle.load(open(HOFile))
else:
    print "Locating all of the hydrogen-oxygen contacts (should have %i).." % (len(Os)*8)
    HODict = defaultdict(list)
    HOCount = 0
    for i, a in enumerate(M.elem):
        if a[0] == 'H':
            for j, b in enumerate(M.elem):
                if b[0] == 'O':
                    if pbcdx(M.xyzs[0][i],M.xyzs[0][j], 2.5) < 2.5:
                        HODict[i].append(j)
                        HOCount += 1
                        print "\rTotal %i found" % HOCount,
    with open(HOFile,"w") as f: pickle.dump(HODict,f)

print "Finding hydrogen-bonding quadruplets (should have %i)" % (len(Os)*2)
HBonds = []
HBondCount = 0
for hi in HODict:
    for hj in HODict:
        if hi < hj and HODict[hi] == HODict[hj]:
            oa = HODict[hi][0]
            ob = HODict[hi][1]
            ha = [hi,hj][np.argmin([pbcdx(M.xyzs[0][oa],M.xyzs[0][hi]),pbcdx(M.xyzs[0][oa],M.xyzs[0][hj])])]
            hb = [hi,hj][np.argmin([pbcdx(M.xyzs[0][ob],M.xyzs[0][hi]),pbcdx(M.xyzs[0][ob],M.xyzs[0][hj])])]
            HBondCount += 1
            
            occa = OccDict[M.elem[ha].lower()] if M.elem[ha].lower() in OccDict else 0.5
            occb = OccDict[M.elem[hb].lower()] if M.elem[hb].lower() in OccDict else 0.5
            if occa + occb < 0.99 or occa + occb > 1.01:
                print "Hydrogens %i and %i should have occupanies that add up to one" % (ha, hb)
                raise

            print "\rIdentified a hydrogen-bonding quadruplet %i-%i-%i-%i (%i total)" % (oa,ha,hb,ob,HBondCount),
            HBonds.append(HBond(oa,ob,ha,hb,occa,occb))

print "Annealing hydrogen positions"
Sys = System(HBonds, Os, M.elem)

NSites0 = defaultdict(int)
for j in M.elem:
    j = j.lower()
    if j in OccDict:
        NSites0[j] += 1

Sys.Anneal()

M1 = M.atom_select(Sys.Select())

NSites1 = defaultdict(int)
for j in M1.elem:
    j = j.lower()
    if j in OccDict:
        NSites1[j] += 1

print
for j in OccDict:
    print "Atom type %s : Ideal occupation % .3f, real occupation % .3f" % (j, OccDict[j], NSites1[j]/NSites0[j])

# Get the first letter only.
M1.elem = [i[0] for i in M1.elem]

Chg = np.array([-0.8 if e[0] == 'O' else 0.4 for e in M1.elem]).reshape(-1,1)
Dip = np.sum((Chg * M1.xyzs[0]) * 4.80245, axis=0)
print "The norm of the dipole moment is % .3f Debye" % np.linalg.norm(Dip),
if np.linalg.norm(Dip) < 1.0:
    print "\x1b[1;95m Ding Ding Ding Ding Ding Ding \x1b[0m"
else:
    print

if np.linalg.norm(Dip) < 10.0:
    M1.comms[0] = "Generated by %s ; dipole moment [% .2f, % .2f, % .2f] D ; lattice vectors %s" % (__file__, Dip[0], Dip[1], Dip[2], format_box(box))
    print
    print "Reordering the atoms for printout"
    Top  = M1.build_topology(Fac=1.3)
    Mols = nx.connected_component_subgraphs(Top)
    M2 = M1.atom_select(np.array(list(itertools.chain(*[list(np.array(i.L())[np.argsort(np.array(i.e()))])[::-1] for i in Mols]))))
    outdir = "output"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    serial = 0
    while True:
        outfnm = os.path.join(outdir,os.path.splitext(os.path.split(fnm)[1])[0]+"_%05i.xyz" % serial)
        if os.path.exists(outfnm):
            serial += 1
        else:
            break
    M2.write(outfnm)
    print "Output written to %s!" % outfnm
else:
    print "Dipole moment is too large, not writing (Sorry, try again!)"
