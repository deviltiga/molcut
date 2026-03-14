#!/usr/bin/python
# Import required modules
from PySide6 import QtCore, QtGui, QtSvg, QtWidgets
import sys
import logging
from warnings import warn
import copy
import csv


import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry.rdGeometry import Point2D, Point3D

# from rdkit.Chem.AllChem import GenerateDepictionMatching3DStructure

from rdeditor.molViewWidget import MolWidget
from rdeditor.templatehandler import TemplateHandler

# from types import *

from rdeditor.ptable import symboltoint

debug = True

"""
训练数据标注&储存工具修改版
"""

# The Molblock editor class
class MolEditWidget(MolWidget):
    def __init__(self, mol=None, parent=None):
        # Also init the super class
        super(MolEditWidget, self).__init__(parent)

        # This sets the window to delete itself when its closed, so it doesn't keep querying the model
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        # Templater handler
        self.templatehandler = TemplateHandler()
        self.sanitize_on_cleanup = True
        self.kekulize_on_cleanup = True

        # Properties
        self._prevmol = None  # For undo
        self.coordlist = None  # SVG coords of the current mols atoms

        # Standard atom, bond and ring types
        self.symboltoint = symboltoint
        self.bondtypes = Chem.rdchem.BondType.names  # A dictionary with all available rdkit bondtypes
        self.available_rings = self.templatehandler.templateslabels  # ["ALI6", "ARO6"]

        # Default actions
        self._action = "Add"
        # self._chemEntityType = "bond"
        # self._chemEntitySubType = self.bondtypes["SINGLE"]
        self.chemEntity = self.bondtypes["SINGLE"]

        # Points to calculate the SVG to coord scaling
        self.points = [Point2D(0, 0), Point2D(1, 1)]

        # Bind signals to slots
        self.finishedDrawing.connect(self.update_coordlist)  # When drawing finished, update coordlist of SVG atoms.

        # Init with a mol if passed at construction
        # if mol != None:
        self.mol = mol


        self.mark_list = []  # 用于存储标记的化学键
        self.mark_atom_list = []  # 用于存储标记的原子


        self._mol = None
        self._canonical_mol = None
        self._needs_canonical_refresh = False
        self.set_mol(mol)

        # 创建 Mark 按钮
        self.mark_button = QtWidgets.QPushButton("Mark", self)
        self.mark_button.clicked.connect(self.toggle_mark_mode)
        self.mark_button.move(10, 10)  # 设置按钮位置
        self.mark_button.show()

        # 创建 Save 按钮
        self.save_button = QtWidgets.QPushButton("Save", self)
        self.save_button.clicked.connect(self.save_marked_bonds)
        self.save_button.move(100, 10)  # 设置按钮位置
        self.save_button.show()

        # 创建 Mark Atom 按钮
        self.mark_atom_button = QtWidgets.QPushButton("Mark Atom", self)
        self.mark_atom_button.clicked.connect(self.toggle_mark_atom_mode)
        self.mark_atom_button.move(10, 50)  # 设置按钮位置
        self.mark_atom_button.show()

        # 创建 Save Atom 按钮
        self.save_atom_button = QtWidgets.QPushButton("Save Atom", self)
        self.save_atom_button.clicked.connect(self.save_marked_atoms)
        self.save_atom_button.move(100, 50)  # 设置按钮位置
        self.save_atom_button.show()



    def toggle_mark_mode(self):
        if self.action != "Mark":
            self.setAction("Mark")
            print("Switched to Mark mode.")
            self.mark_list.clear()
            if self._needs_canonical_refresh:
                self.refresh_canonical_mol()
                self._mol = self._canonical_mol
                self.update_2d_image()
            self.update_mark_highlight()
        else:
            self.setAction("Select")
            print("Exiting Mark mode.")
            self.output_mark_list()

    def toggle_mark_atom_mode(self):
        if self.action != "Mark Atom":
            self.setAction("Mark Atom")
            print("Switched to Mark Atom mode.")
            self.mark_atom_list.clear()
            if self._needs_canonical_refresh:
                self.refresh_canonical_mol()
                self._mol = self._canonical_mol
                self.update_2d_image()
            self.update_mark_atom_highlight()
        else:
            self.setAction("Select")
            print("Exiting Mark Atom mode.")
            self.output_mark_atom_list()

    def handle_mark_bond(self, bond):
        bond_idx = bond.GetIdx()
        if bond_idx not in self.mark_list:
            self.mark_list.append(bond_idx)
            self.update_mark_highlight()

    def handle_mark_atom(self, atom):
        atom_idx = atom.GetIdx()
        if atom_idx not in self.mark_atom_list:
            self.mark_atom_list.append(atom_idx)
            self.logger.debug(f"Marked atom {atom_idx}")
            self.update_mark_atom_highlight()
        else:
            self.mark_atom_list.remove(atom_idx)
            self.logger.debug(f"Unmarked atom {atom_idx}")
            self.update_mark_atom_highlight()

    def update_mark_highlight(self):
        """更新并高亮显示所有标记的化学键"""
        size = self.size()
        self.drawer = rdMolDraw2D.MolDraw2DSVG(int(size.width()), int(size.height()))
        
        if len(self.mark_list) > 0:
            bond_colors = {int(bond_idx): (1.0, 0.0, 0.0) for bond_idx in self.mark_list}
            highlight_bonds = [int(bond_idx) for bond_idx in self.mark_list]

            self.drawer.DrawMolecule(
                self._drawmol,
                highlightAtoms=[],
                highlightBonds=highlight_bonds,
                highlightBondColors=bond_colors,
            )
        else:
            self.drawer.DrawMolecule(self._drawmol)

        self.drawer.FinishDrawing()
        svg = self.drawer.GetDrawingText().replace("svg:", "")
        self.load(QtCore.QByteArray(svg.encode("utf-8")))
        self.update()
        
        self.update_coordlist()



    def update_mark_atom_highlight(self):
        size = self.size()
        self.drawer = rdMolDraw2D.MolDraw2DSVG(int(size.width()), int(size.height()))
        
        if len(self.mark_atom_list) > 0:
            atom_colors = {int(atom_idx): (1.0, 0.0, 0.0) for atom_idx in self.mark_atom_list}
            highlight_atoms = [int(atom_idx) for atom_idx in self.mark_atom_list]

            self._drawmol = rdMolDraw2D.PrepareMolForDrawing(self._mol)
            self.drawer.DrawMolecule(
                self._drawmol,
                highlightAtoms=highlight_atoms,
                highlightAtomColors=atom_colors,
            )
        else:
            self._drawmol = rdMolDraw2D.PrepareMolForDrawing(self._mol)
            self.drawer.DrawMolecule(self._drawmol)

        self.drawer.FinishDrawing()
        svg = self.drawer.GetDrawingText().replace("svg:", "")
        self.load(QtCore.QByteArray(svg.encode("utf-8")))
        self.update()
        
        self.update_coordlist()


    def bond_click(self, bond):
        if self.action == "Mark":
            self.handle_mark_bond(bond)
        else:
            super(MolEditWidget, self).bond_click(bond)



    def save_marked_atoms(self):
        """保存标记的原子到 CSV 文件 SMILES 中的原子顺序"""
        # 获取当前分子的 SMILES 结构
        smiles = Chem.MolToSmiles(self.mol)

        # 获取 SMILES 中的原子映射到 Mol 对象的原子索引的映射
        marked_atoms = [(self.mol.GetAtomWithIdx(idx).GetSymbol(), idx) for idx in self.mark_atom_list]

        # 打开 CSV 文件，如果文件不存在则创建
        with open("marked_atoms.csv", mode="a", newline='') as file:
            writer = csv.writer(file)
            # 如果文件是新建的，写入表头
            if file.tell() == 0:
                writer.writerow(["SMILES", "Marked Atoms"])
            # 写入 SMILES 和标记的原子
            writer.writerow([smiles, marked_atoms])
        
        print(f"Saved {len(self.mark_atom_list)} marked atoms to marked_atoms.csv")


    def output_mark_atom_list(self):
        """输出并保存原子列表"""
        atom_objects = [self.mol.GetAtomWithIdx(idx) for idx in self.mark_atom_list]
        print("Marked Atoms:", [(atom.GetIdx(), atom.GetSymbol()) for atom in atom_objects])

        with open("marked_atoms.txt", "w") as f:
            for atom in atom_objects:
                f.write(f"Atom {atom.GetIdx()} - Symbol: {atom.GetSymbol()}\n")












    @property
    def mol(self):
        return self._mol

    @mol.setter
    def mol(self, mol):
        self.set_mol(mol)

    def set_mol(self, mol):
        if mol is None:
            mol = Chem.MolFromSmiles("")
        
        if mol != self._mol:
            self._prevmol = Chem.Mol(self._mol) if self._mol is not None else None
            self._mol = Chem.Mol(mol)
            self._needs_canonical_refresh = True
            
            # 确保有2D坐标
            if self._mol.GetNumConformers() == 0:
                AllChem.Compute2DCoords(self._mol)
            
            # 更新UI
            self.molChanged.emit()
            self.update_coordlist()

    def refresh_canonical_mol(self):
        if self._mol and self._needs_canonical_refresh:
            # 获取原始 SMILES，保持组分顺序
            original_smiles = Chem.MolToSmiles(self._mol, isomericSmiles=True)
            
            # 分割多组分 SMILES
            components = original_smiles.split('.')
            
            # 为每个组分创建带有原子映射的 SMILES
            mapped_components = []
            atom_counter = 1
            for component in components:
                mol = Chem.MolFromSmiles(component)
                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(atom_counter)
                    atom_counter += 1
                mapped_components.append(Chem.MolToSmiles(mol, isomericSmiles=True))
            
            # 重新组合带有原子映射的 SMILES
            mapped_smiles = '.'.join(mapped_components)
            
            # 从带有原子映射的 SMILES 创建新的分子
            self._canonical_mol = Chem.MolFromSmiles(mapped_smiles)
            
            if self._canonical_mol is None:
                self.logger.error(f"Failed to create canonical molecule from SMILES: {mapped_smiles}")
                return

            # 移除原子映射号，但保持原子顺序
            for atom in self._canonical_mol.GetAtoms():
                atom.SetAtomMapNum(0)

            # 复制属性
            for prop in self._mol.GetPropNames():
                self._canonical_mol.SetProp(prop, self._mol.GetProp(prop))
            
            # 生成2D坐标
            AllChem.Compute2DCoords(self._canonical_mol)
            
            # Sanitize and update properties
            try:
                Chem.SanitizeMol(self._canonical_mol)
            except:
                self.logger.warning("Failed to sanitize canonical molecule")
            
            self._canonical_mol.UpdatePropertyCache(strict=False)
            
            self._needs_canonical_refresh = False
            
            # 更新原始分子以匹配规范化的分子
            self._mol = Chem.Mol(self._canonical_mol)

    def toggle_mark_mode(self):
        if self.action != "Mark":
            self.setAction("Mark")
            print("Switched to Mark mode.")
            self.mark_list.clear()
            if self._needs_canonical_refresh:
                self.refresh_canonical_mol()
                self._mol = self._canonical_mol
                self.update_2d_image()
            self.update_mark_highlight()
        else:
            self.setAction("Select")
            print("Exiting Mark mode.")
            self.output_mark_list()

    def update_2d_image(self):
        if self._mol:
            AllChem.Compute2DCoords(self._mol)
            self.molChanged.emit()
            self.update_coordlist()






    def save_marked_bonds(self):
        """保存标记的化学键到 CSV 文件 SMILES 中的原子顺序"""
        # 获取当前分子的 SMILES 结构
        smiles = Chem.MolToSmiles(self.mol)

        # 获取 SMILES 中的原子映射到 Mol 对象的原子索引的映射
        marked_bonds = [self.mol.GetBondWithIdx(idx) for idx in self.mark_list]

        # 打开 CSV 文件，如果文件不存在则创建 可以改名字
        with open("marked_bonds.csv", mode="a", newline='') as file:
            writer = csv.writer(file)
            # 如果文件是新建的，写入表头
            if file.tell() == 0:
                writer.writerow(["SMILES", "Marked Bonds"])
            # 写入 SMILES 和标记的化学键
            writer.writerow([smiles, [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in marked_bonds]])
        
        print(f"Saved {len(self.mark_list)} marked bonds to marked_bonds.csv")

    """
    def toggle_mark_mode(self):
        
        if self.action != "Mark":
            self.setAction("Mark")
            print("Switched to Mark mode.")
            self.mark_list.clear()  # 清空之前的 mark 列表
        else:
            self.setAction("Select")  # 恢复到 Select 模式
            print("Exiting Mark mode.")
            self.output_mark_list()  # 调用输出函数
    """

    def output_mark_list(self):
        """输出并保存化学键列表"""
        # 这里假设 self.mark_list 存储的是化学键的索引
        bond_objects = [self.mol.GetBondWithIdx(idx) for idx in self.mark_list]
        print("Marked Bonds:", [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in bond_objects])

        with open("marked_bonds.txt", "w") as f:
            for bond in bond_objects:
                f.write(f"Bond between atoms {bond.GetBeginAtomIdx()} and {bond.GetEndAtomIdx()}\n")

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            clicked = self.get_molobject(event)
            if isinstance(clicked, Chem.rdchem.Bond):
                self.bond_click(clicked)
            else:
                super(MolEditWidget, self).mousePressEvent(event)

    def bond_click(self, bond):
        if self.action == "Mark":
            self.handle_mark_bond(bond)
        else:
            super(MolEditWidget, self).bond_click(bond)
    """
    def handle_mark_bond(self, bond):
        
        bond_idx = bond.GetIdx()
        if bond_idx not in self.mark_list:
            self.mark_list.append(bond_idx)
            self.update_mark_highlight()
    """












    # Getters and Setters for properties
    actionChanged = QtCore.Signal(name="actionChanged")

    @property
    def action(self):
        return self._action

    @action.setter
    def action(self, actionname):
        if actionname != self.action:
            self._action = actionname
            self.actionChanged.emit()

    def setAction(self, actionname):
        self.action = actionname

    # bondTypeChanged = QtCore.Signal(name="bondTypeChanged")

    # chemEntityTypeChanged = QtCore.Signal(name="chemEntityTypeChanged")

    @property
    def chemEntity(self):
        return self._chementity

    @chemEntity.setter
    def chemEntity(self, chementity):
        if isinstance(chementity, Chem.rdchem.BondType):  # Bondtypes are also ints, but ints are not BondTypes
            self.setBond(chementity)
        elif isinstance(chementity, int):
            self.setAtom(chementity)
        elif isinstance(chementity, str):
            if chementity in self.bondtypes:
                self.setBond(chementity)
            elif chementity in self.available_rings:
                self.setRing(chementity)
            elif chementity in symboltoint.keys():
                self.setAtom(chementity)
            else:
                self.logger.error(f"Unknown string entity type with value {chementity}")
                return
        else:
            self.logger.error(f"Unknown type {type(chementity)}")
            return
        # self.logger.debug(f"ChemEntity set for {chementity} of type {type(chementity)}")

    def setChemEntity(self, chementity):
        self.chemEntity = chementity

    # Readonly, inferred from chemEntity
    @property
    def chemEntityType(self):
        return self._chementitytype

    @property
    def bondtype(self):
        warn(
            ".bondtype has been deprecated, in favor of .chemEntityType and setter .chemEntity.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._bondtype

    @property
    def ringtype(self):
        warn(
            ".ringtype has been deprecated, in favor of .chemEntityType and setter .chemEntity.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._ringtype

    @bondtype.setter
    def bondtype(self, bondtype):
        warn(
            ".bondtype has been deprecated, in favor of .chemEntityType and setter .chemEntity.",
            DeprecationWarning,
            stacklevel=2,
        )
        if bondtype != self.bondtype:
            self._bondtype = bondtype
            # self.bondTypeChanged.emit()

    @ringtype.setter
    def ringtype(self, ringtype):
        warn(
            ".ringtype has been deprecated, in favor of .chemEntityType and setter .chemEntity.",
            DeprecationWarning,
            stacklevel=2,
        )
        if ringtype != self.ringtype:
            self._ringtype = ringtype

    def setRing(self, ringtype):
        if ringtype in self.available_rings:
            self._chementitytype = "ring"
            self._chementity = ringtype
        else:
            self.logger.error(f"Currently only {self.available_rings} are supported.")

    def setBond(self, bondtype):
        if isinstance(bondtype, Chem.rdchem.BondType):
            self._chementitytype = "bond"
            self._chementity = bondtype

        elif isinstance(bondtype, str):
            assert bondtype in self.bondtypes.keys(), "Bondtype %s not known" % bondtype
            self._chementitytype = "bond"
            self._chementity = self.bondtypes[bondtype]
        else:
            self.logger.error("Bondtype must be string or rdchem.BondType, not %s" % type(bondtype))

    @property
    def atomtype(self):
        warn(
            ".atomtype has been deprecated, in favor of .chemEntityType and setter .chemEntity.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._atomtype

    @atomtype.setter
    def atomtype(self, atomtype):
        warn(
            ".atomtype has been deprecated, in favor of .chemEntityType and setter .chemEntity.",
            DeprecationWarning,
            stacklevel=2,
        )
        if atomtype != self.atomtype:
            self._atomtype = atomtype

    def setAtom(self, atomtype):
        self.logger.debug("Setting atomtype selection to %s" % atomtype)
        if atomtype in self.symboltoint.keys():
            self.logger.debug("Atomtype found in keys")
            # self.atomtype = self.symboltoint[atomtype]
            self._chementitytype = "atom"
            self._chementity = self.symboltoint[atomtype]
        elif isinstance(atomtype, int):
            if atomtype in self.symboltoint.values():
                self._chementitytype = "atom"
                self._chementity = atomtype
            else:
                self.logger.error(f"Atom number {atomtype} not known.")
        else:
            self.logger.error("Atomtype must be string or integer, not %s" % type(atomtype))

    # Function to translate from SVG coords to atom coords using scaling calculated from atomcoords (0,0) and (1,1)
    # Returns rdkit Point2D
    def SVG_to_coord(self, x_svg, y_svg):
        if self.drawer is not None:
            scale0 = self.drawer.GetDrawCoords(self.points[0])
            scale1 = self.drawer.GetDrawCoords(self.points[1])

            ax = scale1.x - scale0.x
            bx = scale0.x

            ay = scale1.y - scale0.y
            by = scale0.y

            return Point2D((x_svg - bx) / ax, (y_svg - by) / ay)
        else:
            return Point2D(0.0, 0.0)

    def update_coordlist(self):
        if self.mol is not None:
            self.coordlist = np.array([list(self.drawer.GetDrawCoords(i)) for i in range(self.mol.GetNumAtoms())])
            self.logger.debug("Current coordlist:\n%s" % self.coordlist)
        else:
            self.coordlist = None

    def get_nearest_atom(self, x_svg, y_svg):
        if self.mol is not None and self.mol.GetNumAtoms() > 0:
            atomsvgcoords = np.array([x_svg, y_svg])
            # find distance, https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
            deltas = self.coordlist - atomsvgcoords
            dist_2 = np.einsum("ij,ij->i", deltas, deltas)
            min_idx = np.argmin(dist_2)
            return min_idx, dist_2[min_idx] ** 0.5
        else:
            return None, 1e10  # Return ridicilous long distance so that its not chosen

    def get_nearest_bond(self, x_svg, y_svg):
        if self.mol is not None and len(self.mol.GetBonds()) > 0:
            bondlist = []
            for bond in self.mol.GetBonds():
                bi = bond.GetBeginAtomIdx()
                ei = bond.GetEndAtomIdx()
                avgcoords = np.mean(self.coordlist[[bi, ei]], axis=0)
                bondlist.append(avgcoords)

            bondlist = np.array(bondlist)
            # if not bondlist:  # If there's no bond
            #     return None, 1e10
            atomsvgcoords = np.array([x_svg, y_svg])
            deltas = bondlist - atomsvgcoords
            dist_2 = np.einsum("ij,ij->i", deltas, deltas)
            min_idx = np.argmin(dist_2)
            return min_idx, dist_2[min_idx] ** 0.5
        else:
            return None, 1e10  # Return ridicilous long distance so that its not chosen

    # Function that translates coodinates from an event into a molobject
    def get_molobject(self, event):
        # Recalculate to SVG coords
        viewbox = self.renderer().viewBox()
        size = self.size()

        x = event.pos().x()
        y = event.pos().y()
        # Rescale, divide by the size of the widget, multiply by the size of the viewbox + offset.
        x_svg = float(x) / size.width() * viewbox.width() + viewbox.left()
        y_svg = float(y) / size.height() * viewbox.height() + viewbox.top()
        self.logger.debug("SVG_coords:\t%s\t%s" % (x_svg, y_svg))
        # Identify Nearest atomindex
        atom_idx, atom_dist = self.get_nearest_atom(x_svg, y_svg)
        bond_idx, bond_dist = self.get_nearest_bond(x_svg, y_svg)
        self.logger.debug("Distances to atom %0.2F, bond %0.2F" % (atom_dist, bond_dist))
        # If not below a given threshold, then it was not clicked
        if min([atom_dist, bond_dist]) < 14.0:
            if atom_dist < bond_dist:
                return self.mol.GetAtomWithIdx(int(atom_idx))
            else:
                return self.mol.GetBondWithIdx(int(bond_idx))
        else:
            # Translate SVG to Coords
            return self.SVG_to_coord(x_svg, y_svg)

    def mousePressEvent(self, event):
        if event.button() is QtCore.Qt.LeftButton:
            clicked = self.get_molobject(event)
            if isinstance(clicked, Chem.rdchem.Atom):
                self.logger.debug(
                    "You clicked atom %i, with atomic number %i" % (clicked.GetIdx(), clicked.GetAtomicNum())
                )
                # Call the atom_click function
                self.atom_click(clicked)
                # self.add_atom(self.pen, clicked)
            elif isinstance(clicked, Chem.rdchem.Bond):
                self.logger.debug("You clicked bond %i with type %s" % (clicked.GetIdx(), clicked.GetBondType()))
                self.bond_click(clicked)
            elif isinstance(clicked, Point2D):
                self.logger.debug("Canvas Click")
                self.canvas_click(clicked)
            else:
                self.logger.error(f"Clicked entity, {clicked} of unknown type {type(clicked)}")

    # Lookup tables to relate actions to context type with action type #TODO more clean to use Dictionaries??
    def atom_click(self, atom):
        if self.action == "Mark Atom":
            self.handle_mark_atom(atom)
        elif self.action == "Add":
            self.add_to_atom(atom)
        elif self.action == "Remove":
            self.remove_atom(atom)
        elif self.action == "Select":
            self.select_atom_add(atom)
        elif self.action == "Replace":
            self.replace_on_atom(atom)
        elif self.action == "Add Bond":
            self.add_bond(atom)
        elif self.action == "Increase Charge":
            self.increase_charge(atom)
        elif self.action == "Decrease Charge":
            self.decrease_charge(atom)
        elif self.action == "RStoggle":
            self.toogleRS(atom)
        else:
            self.logger.warning(f"The combination of Atom click and Action {self.action} undefined")

    # 添加了mark模式
    def bond_click(self, bond):
        if self.action == "Add":
            self.add_to_bond(bond)
        elif self.action == "Add Bond":
            self.replace_bond(bond)
        elif self.action == "Remove":
            self.remove_bond(bond)
        elif self.action == "Select":
            self.select_bond(bond)
        elif self.action == "Replace":
            self.replace_on_bond(bond)
        elif self.action == "EZtoggle":
            self.toogleEZ(bond)
        elif self.action == "Mark":
            self.handle_mark_bond(bond)
        else:
            self.logger.warning("The combination of Bond click and Action %s undefined" % self.action)


    def canvas_click(self, point):
        if self.action == "Add":
            self.add_canvas_entity(point)

        elif self.action == "Select":
            # Click on canvas
            # Unselect any selected
            if len(self.selectedAtoms) > 0:
                self.clearAtomSelection()
        else:
            self.logger.warning("The combination of Canvas click and Action %s undefined" % self.action)

    def add_to_atom(self, atom):
        if self.chemEntityType == "atom":
            self.add_atom_to_atom(atom)
        if self.chemEntityType == "ring":
            self.add_ring_to_atom(atom)
        if self.chemEntityType == "bond":
            self.add_bond_to_atom(atom)

    def add_atom_to_atom(self, atom):
        rwmol = Chem.rdchem.RWMol(self.mol)
        newatom = Chem.rdchem.Atom(self.chemEntity)
        newidx = rwmol.AddAtom(newatom)
        newbond = rwmol.AddBond(atom.GetIdx(), newidx, Chem.rdchem.BondType.SINGLE)
        self.mol = rwmol

    def add_bond_to_atom(self, atom):
        rwmol = Chem.rdchem.RWMol(self.mol)
        newatom = Chem.rdchem.Atom(6)
        newidx = rwmol.AddAtom(newatom)
        newbond = rwmol.AddBond(atom.GetIdx(), newidx, order=self.chemEntity)
        self.mol = rwmol

    def add_ring_to_atom(self, atom):
        mol = self.templatehandler.apply_template_to_atom(atom, self.chemEntity)
        self.mol = mol

    def add_to_bond(self, bond):
        if self.chemEntityType == "atom":
            self.toggle_bond(bond)
        if self.chemEntityType == "ring":
            self.add_ring_to_bond(bond)
        if self.chemEntityType == "bond":
            self.replace_bond(bond)

    def add_ring_to_bond(self, bond):
        mol = self.templatehandler.apply_template_to_bond(bond, self.chemEntity)
        self.mol = mol

    def add_canvas_entity(self, point):
        if self.chemEntityType == "atom":
            self.add_canvas_atom(point)
        if self.chemEntityType == "ring":
            self.add_canvas_ring(point)
        if self.chemEntityType == "bond":
            self.add_canvas_bond(point)

    def add_canvas_atom(self, point):
        rwmol = Chem.rdchem.RWMol(self.mol)
        if rwmol.GetNumAtoms() == 0:
            point.x = 0.0
            point.y = 0.0
        newatom = Chem.rdchem.Atom(self.chemEntity)
        newidx = rwmol.AddAtom(newatom)
        # This should only trigger if we have an empty canvas
        if not rwmol.GetNumConformers():
            rdDepictor.Compute2DCoords(rwmol)
        conf = rwmol.GetConformer(0)
        p3 = Point3D(point.x, point.y, 0)
        conf.SetAtomPosition(newidx, p3)
        self.mol = rwmol

    def add_canvas_bond(self, point):
        rwmol = Chem.rdchem.RWMol(self.mol)
        if rwmol.GetNumAtoms() == 0:
            point.x = 0.0
            point.y = 0.0

        atom_0 = rwmol.AddAtom(Chem.rdchem.Atom(6))
        atom_1 = rwmol.AddAtom(Chem.rdchem.Atom(6))
        print(self.chemEntity)
        newidx = rwmol.AddBond(atom_0, atom_1, order=self.chemEntity)

        # This should only trigger if we have an empty canvas
        if not rwmol.GetNumConformers():
            rdDepictor.Compute2DCoords(rwmol)
        conf = rwmol.GetConformer(0)
        p3 = Point3D(point.x, point.y, 0)
        conf.SetAtomPosition(self.mol.GetNumAtoms(), p3)
        self.mol = rwmol

    def add_canvas_ring(self, point):
        mol = self.templatehandler.apply_template_to_canvas(self.mol, point, self.chemEntity)
        self.mol = mol

    def remove_atom(self, atom):
        rwmol = Chem.rdchem.RWMol(self.mol)
        rwmol.RemoveAtom(atom.GetIdx())
        self.clearAtomSelection()  # Removing atoms updates Idx'es
        self.mol = rwmol

    def select_atom(self, atom):
        self.selectAtom(atom.GetIdx())
        # TODO make an unselect atom function

    def select_atom_add(self, atom):
        selidx = atom.GetIdx()
        if selidx in self._selectedAtoms:
            self.unselectAtom(selidx)
        else:
            self.selectAtomAdd(selidx)

    def replace_on_atom(self, atom):
        if self.chemEntityType == "atom":
            self.replace_atom(atom)
        else:
            pass

    def replace_atom(self, atom):
        rwmol = Chem.rdchem.RWMol(self.mol)
        newatom = Chem.rdchem.Atom(self.chemEntity)
        rwmol.ReplaceAtom(atom.GetIdx(), newatom)
        self.mol = rwmol

    # Double step action
    def add_bond(self, atom):
        if len(self.selectedAtoms) > 0:
            selected = self.selectedAtoms[-1]
            rwmol = Chem.rdchem.RWMol(self.mol)
            neighborIdx = [atm.GetIdx() for atm in self.mol.GetAtomWithIdx(selected).GetNeighbors()]
            if atom.GetIdx() not in neighborIdx:  # check if bond already exists
                bondType = self.chemEntity if self.chemEntityType == "bond" else Chem.rdchem.BondType.SINGLE
                rwmol.AddBond(selected, atom.GetIdx(), order=bondType)
            self.mol = rwmol
            self.selectedAtoms = []
        else:
            self.select_atom(atom)

    def toogleRS(self, atom):
        self.backupMol()
        # atom = self._mol.GetAtomWithIdx(atom.GetIdx())
        stereotype = atom.GetChiralTag()
        self.logger.debug("Current stereotype of clicked atom %s" % stereotype)
        stereotypes = [
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            # Chem.rdchem.ChiralType.CHI_OTHER, this one doesn't show a wiggly bond
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        ]
        newidx = np.argmax(np.array(stereotypes) == stereotype) + 1
        atom.SetChiralTag(stereotypes[newidx])
        self.logger.debug("New stereotype set to %s" % atom.GetChiralTag())
        # rdDepictor.Compute2DCoords(self._mol)
        # self._mol.ClearComputedProps()
        self._mol.UpdatePropertyCache(strict=False)
        rdDepictor.Compute2DCoords(self._mol)
        self.molChanged.emit()

    def assert_stereo_atoms(self, bond):
        if len(bond.GetStereoAtoms()) == 0:
            # get atoms and idx's of bond
            bondatoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
            bondidx = [atom.GetIdx() for atom in bondatoms]

            # Figure out the atom idx's of the neigbor atoms, that are NOT the other end of the bond
            stereoatoms = []
            for bondatom in bondatoms:
                neighboridxs = [atom.GetIdx() for atom in bondatom.GetNeighbors()]
                neighboridx = [idx for idx in neighboridxs if idx not in bondidx][0]
                stereoatoms.append(neighboridx)
            # Set the bondstereoatoms
            bond.SetStereoAtoms(stereoatoms[0], stereoatoms[1])
            self.logger.debug(f"Setting StereoAtoms to {stereoatoms}")
        else:
            pass

    def assign_stereo_atoms(self, mol: Chem.Mol):
        self.logger.debug("Identifying stereo atoms")
        mol_copy = copy.deepcopy(mol)
        Chem.SanitizeMol(mol_copy, sanitizeOps=Chem.rdmolops.SanitizeFlags.SANITIZE_SYMMRINGS)
        Chem.rdmolops.FindPotentialStereoBonds(mol_copy, cleanIt=True)
        for i, bond in enumerate(mol_copy.GetBonds()):
            stereoatoms = list(
                set(bond.GetStereoAtoms())
            )  # Is FindPotentialStereoBonds are run successively, the list is simply expanded.
            if stereoatoms:
                try:
                    mol.GetBondWithIdx(i).SetStereoAtoms(stereoatoms[0], stereoatoms[1])
                except RuntimeError:
                    mol.GetBondWithIdx(i).SetStereoAtoms(
                        stereoatoms[1], stereoatoms[0]
                    )  # Not sure why this can get the wrong way. Seem to now work correctly for Absisic Acid

    def toogleEZ(self, bond: Chem.Bond):
        self.backupMol()

        stereotype = bond.GetStereo()  # TODO, when editing the molecule, we could change the CIP rules?
        # so stereo assignment need to be updated on other edits as well?
        self.logger.debug("Current stereotype of clicked atom %s" % stereotype)
        self.logger.debug(f"StereoAtoms are {list(bond.GetStereoAtoms())}")
        self.logger.debug(f"Bond properties are {bond.GetPropsAsDict(includePrivate=True, includeComputed=True)}")

        self.assign_stereo_atoms(self._mol)  # TODO, make something that ONLY works on a single bond?

        stereocycler = {
            Chem.rdchem.BondStereo.STEREONONE: Chem.rdchem.BondStereo.STEREOTRANS,
            Chem.rdchem.BondStereo.STEREOE: Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS: Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOZ: Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOCIS: Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOANY: Chem.rdchem.BondStereo.STEREONONE,
        }

        newstereotype = stereocycler[stereotype]
        bond.SetStereo(newstereotype)

        self.logger.debug("New stereotype set to %s" % bond.GetStereo())
        self.logger.debug(f"StereoAtoms are {list(bond.GetStereoAtoms())}")
        self.logger.debug(f"Bond properties are {bond.GetPropsAsDict(includePrivate=True, includeComputed=True)}")

        self.logger.debug(f"StereoAtoms are {list(bond.GetStereoAtoms())}")
        self.logger.debug(f"Bond properties are {bond.GetPropsAsDict(includePrivate=True, includeComputed=True)}")

        self.molChanged.emit()

    # Bond actions
    def toggle_bond(self, bond):
        self.backupMol()
        bondtype = bond.GetBondType()
        bondtypes = [
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
        ]
        # Find the next type in the list based on current
        # If current is not in list? Then it selects the first and add 1 => SINGLE
        newidx = np.argmax(np.array(bondtypes) == bondtype) + 1
        newtype = bondtypes[newidx]
        bond.SetBondType(newtype)
        self.molChanged.emit()

    def replace_on_bond(self, bond):
        if self.chemEntityType == "atom":
            self.toggle_bond(bond)
        if self.chemEntityType == "ring":
            self.toggle_bond(bond)
        if self.chemEntityType == "bond":
            self.replace_bond(bond)

    def replace_bond(self, bond):
        self.backupMol()
        self.logger.debug("Replacing bond %s" % bond)
        bond.SetBondType(self.chemEntity)
        self.molChanged.emit()

    # self.remove_bond(bond)
    def remove_bond(self, bond):
        rwmol = Chem.rdchem.RWMol(self.mol)
        rwmol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        self.mol = rwmol

    def increase_charge(self, atom):
        self.backupMol()
        atom.SetFormalCharge(atom.GetFormalCharge() + 1)
        self.molChanged.emit()

    def decrease_charge(self, atom):
        self.backupMol()
        atom.SetFormalCharge(atom.GetFormalCharge() - 1)
        self.molChanged.emit()

    # self.select_bond(bond)
    def select_bond(self, bond):
        self.logger.debug("Select_bond not implemented")  # TODO

    def undo(self):
        self.mol = self._prevmol

    def backupMol(self):
        self._prevmol = copy.deepcopy(self.mol)

    def cleanup_mol(self):
        mol = copy.deepcopy(self.mol)
        if self.sanitize_on_cleanup:
            Chem.SanitizeMol(mol)
        if self.kekulize_on_cleanup:
            Chem.Kekulize(mol)
        # if Chem.MolToCXSmiles(self.mol) != Chem.MolToCXSmiles(mol):
        self.mol = mol


if __name__ == "__main__":
    mol = Chem.MolFromSmiles("CCN(C)C1CCCCC1S")
    rdDepictor.Compute2DCoords(mol)
    myApp = QtWidgets.QApplication(sys.argv)
    molblockview = MolWidget(mol)
    molblockview.show()
    myApp.exec()
