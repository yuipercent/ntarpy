import pygame
import numpy
import ast
from pathlib import Path
import random
from math import *
import os
from copy import deepcopy
import numba

global ReScaleCoef
pygame.init()
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
windowHeight = str(pygame.display.Info())
windowLength = int(windowHeight[windowHeight.index("current_w") + 12 : windowHeight.index(",",windowHeight.index("current_w") + 12)])
windowHeight = int(windowHeight[windowHeight.index("current_h") + 12 : windowHeight.index("\n",windowHeight.index("current_w") + 12)])
location = str(Path.cwd()).replace("\\Utils","")
Rooms = {}
Entities = []
SubEntities = []
Constraints = []
laws = [{"formula":"-9.81*weight","axis":"x"}]
AEntities = {}
radiantcoef = pi/180
palettes = {}
ReScaleCoef = 3
RequestScheme = numpy.ndarray((640,360,2),numpy.float32)

@numba.njit(cache=True,nogil=True,fastmath=True)
def packFOV(array : numpy.ndarray,angleh,anglev,toadd,shape):
    for dim1 in range(0,shape[0]):
        for dim2 in range(0,shape[1]):
            array[dim1][dim2][0] = ((dim1-shape[0]/2)*(angleh/(shape[0]/2)))+0.000000001
            array[dim1][dim2][1] = ((dim2-shape[1]/2)*(anglev/(shape[1]/2)))+0.000000001
    return array

class rnk():
    """A list. Elements within are ordered based on the number they are associated with."""
    def __init__(self):
        self.items = []
        self.minvalue = 0
        self.maxvalue = 0
        self.__nbitem = 0
        self.__lastadded = None
    
    def add(self,obj,value):
        self.items.append(value),self.items.append(obj)
        self.__update()
        self.__nbitem += 1
        self.__lastadded = obj
    
    def __update(self):
        a = []
        for l in self.items:
            if type(l) == int:
                a.append(l)
        if not len(self.items)==0:
            self.maxvalue = max(a)
            self.minvalue = min(a)
    
    def remove(self,obj):
        try:
            a = self.items.index(obj)
            self.items = self.items[:a-1]+self.items[a+1:]
            self.__nbitem -= 1
        except ValueError:
            pass
        except IndexError:
            self.items = self.items[:a-1]
        self.__update()

    def returnOrderedlist(self,i = None):
        a = []
        b = self.maxvalue
        l = self.items[:]
        while True:
            try:
                a.append(l[l.index(b)+1])
                l[l.index(b)] = None
            except ValueError:
                b -= 1
            if b < self.minvalue:
                return a

    def returnOrderedlistInv(self,i = None):
        a = []
        b = self.minvalue
        if b < i:
            b = i
        l = self.items[:]
        while True:
            try:
                a.append(l[l.index(b)+1])
                l[l.index(b)] = None
            except ValueError:
                b += 1
            if b > self.maxvalue:
                return a
    
    def __str__(self):
        return str(self.items)

    def changeRnk(self,obj,v):
        try:
            self.items[self.items.index(obj)-1] = v
        except ValueError:
            return -1
        self.__update()
    
    def getrnklen(self,rnk):
        b = 0
        for l in self.items:
            if l == rnk:
                b += 1
        return b
    
    def len(self):
        return self.__nbitem
    
    def returnLastAdded(self):
        return self.__lastadded
    
    def __getitem__(self,index):
        if index*2+1 < len(self.items)*-1:
            index += 1
        return self.items[index*2+1]
    
    def getIndex(self,obj):
        return self.items.index(obj)//2-1

def data_merge(obj,dict_ : dict):
    k = list(dict_.keys())
    v = list(dict_.values())
    for kl in range(0,len(k)):
        obj[k[kl]] = v[kl]

def convert_palette(obj_):
    b = []
    for l in obj_:
        sequence = []
        limit = int(l["limit"][1:])
        u0 = l["u0"]
        start = l["from"]
        to = l["to"]
        un = u0
        n = 0
        for n in range(0,limit):
            un = round(eval(l["un+1"]),4)
            sequence.append(un)
        coef = (1/max(sequence))
        comp = min(sequence)*(1/max(sequence))
        for n in range(0,limit):
            sequence[n] = sequence[n]*coef-comp
        comp = max(sequence)
        for n in range(1,limit):
            sequence[n] = round(sequence[n]*(1/comp),3)
        for n in range(0,limit):
            sequence[n] = numpy.asarray((int(start[0]*(1-sequence[n])+to[0]*sequence[n]),int(start[1]*(1-sequence[n])+to[1]*sequence[n]),int(start[2]*(1-sequence[n])+to[2]*sequence[n])),dtype=numpy.uint8)
        b.append(tuple(sequence))
    return tuple(b)
        
class Region():
    def __init__(self,nbt_):
        data_merge(self,nbt_)

    def __getitem__(self,item_):
        return getattr(self,item_)
    
    def __setitem__(self,item_,g):
        setattr(self,item_,g)

class LoadedRoom():
    def __init__(self):
        pass
    def __getitem__(self,item_):
        return getattr(self,item_)
    
    def __setitem__(self,item_,g):
        setattr(self,item_,g)

class RoomModel():
    global Rooms
    def __init__(self,nbt_ : dict = {}):
        self.roomLayers = []
        self.Dimensions = [0,0]
        data_merge(self,nbt_)
    
    def addRoomLayer(self,toadd):
        self.roomLayers.append(toadd.convert())

    def __getitem__(self,item_):
        return getattr(self,item_)
    
    def __setitem__(self,item_,g):
        setattr(self,item_,g)
    
    def returnNBT(self):
        return {"roomLayers":[roomLayer(self.roomLayers[t].returnNBT()) for t in range(0,len(self.roomLayers))],"Dimensions":self.Dimensions[:]}

class LoadingRoom(RoomModel):
    def __init__(self,parent : RoomModel):
        a = parent.returnNBT()
        super().__init__(a)
        data_merge(self,a)
        self.Progression = 0
        self.maxProg = len(self.roomLayers)
        self.cProg = [0,0,0]
    
    def continue_conversion(self):
        b = []
        for k in self.roomLayers:
            b.append(k.convert())
        b.reverse()
        toreturn = []
        for k in range(0,3):                            #Room0-2
            p = numpy.ndarray(17,dtype=numpy.int32)
            for l in range(0,len(b)):
                p[l] = b[l][k]
            toreturn.append(p)
        for k in range(0,2):                            #Room3-4
            p = numpy.ndarray(17,dtype=numpy.uint16)
            for l in range(0,len(b)):
                p[l] = b[l][k+3]
            toreturn.append(p)
        p = []
        for l in range(0,len(b)):                       #Room5
            p.append(b[l][5])
        for l in range(0,17-len(b)):                       #Room5
            p.append(numpy.ndarray((5,5,8),dtype=numpy.int8))
        toreturn.append(tuple(p))
        p = numpy.ndarray((17,4,9,255,3),dtype=numpy.uint8)
        for room in range(0,len(b)):                    #Room6
            for palette in range(0,len(b[room][6])):
                for texturetype in range(0,len(b[room][6][palette])):
                    for set in range(0,len(b[room][6][palette][texturetype])):
                        p[room][palette][texturetype][set] = b[room][6][palette][texturetype][set]
        toreturn.append(p)
        toreturn.append(numpy.full(17,True,dtype=bool))
        p = numpy.full(17,False,dtype=bool)
        for l in range(0,len(b)):
            p[l] = True
        toreturn.append(p)
        return toreturn

class roomLayer():
    def __init__(self,nbt_):
        self.Position = [0,0]
        self.Dimensions = [0,0]
        self.Palettes = [palettes["Classic"],]
        data_merge(self,nbt_)

    def __getitem__(self,item_):
        if not type(item_) == int:
            return getattr(self,item_)
        else:
            return self.Graphics[item_]
    
    def __setitem__(self,item_,g):
        setattr(self,item_,g)
    
    def len(self):
        return len(self.Graphics)

    def returnNBT(self):
        return {"Position":self.Position[:],"Dimensions":self.Dimensions[:],"Layer":self.Layer,"Properties":self.Properties}
    
    def convert(self):
        """Retourne une forme convertie de la layer : (Posx,Posy,Posz,Dimx,Dimy,Array4d layer,properties,palettearray,palettes,isEntityLayer)"""
        return (float(self.Position[0]),float(self.Position[1]),float(self.Layer*16),self.Dimensions[0],self.Dimensions[1],self.Properties,self.Palettes,True)

class Entity():
    global Entities
    def __init__(self,nbt_ : dict = {}):
        self.Location, self.SubEntities, self.Origin, self.Name, self.Layer = [0,0,0], {}, None, "Badis mon bebouuuu", 0
        self.Constraints = []
        self.Arguments = {}
        data_merge(self,nbt_)

    def __getitem__(self,item_):
        return getattr(self,item_)
    
    def __setitem__(self,item_,g):
        setattr(self,item_,g)

class Formula():
    global letters
    def __init__(self,f : str):
        self.Formula = []
        n = ["0","1","2","3","4","5","6","7","8","9","."]
        f += "$"
        while not len(f) == 0:
            c = f[0]
            if c in n:
                for t in range(0,len(f)):
                    if f[t] not in n:
                        self.Formula.append(f[0:t])
                        f = f[t-1:]
                        break
            elif c in letters:
                if f[0:3] == "cos":
                    self.Formula.append(cos)
                    f = f[2:]
                elif f[0:3] == "sin":
                    self.Formula.append(sin)
                    f = f[2:]
                elif f[0:3] == "tan":
                    self.Formula.append(tan)
                    f = f[2:]
                elif f[0:3] == "abs":
                    self.Formula.append(abs)
                    f = f[2:]
                elif f[0:4] == "asin":
                    self.Formula.append(asin)
                    f = f[3:]
                elif f[0:4] == "acos":
                    self.Formula.append(acos)
                    f = f[3:]
                elif f[0:4] == "atan":
                    self.Formula.append(atan)
                    f = f[3:]
                else:
                    if not len(self.Formula) == 0:
                        if type(self.Formula[-1]) == float:
                            self.Formula.append("*")
                    self.Formula.append(c)
            elif c in ["*","-","+","/","(",")","%","^"]:
                self.Formula.append(c)
            f = f[1:]
        
    def __str__(self):
        return "".join(self.Formula)

    def calc_for(self,a):
        rlist = self.Formula[:]
        if type(a) == dict:
            for k in range(0,len(rlist)):
                if rlist[k] in letters:
                    try:
                        rlist[k] = str(a[rlist[k]])
                    except KeyError:
                        pass
        elif type(a) == float or type(a) == int:
            for k in range(0,len(rlist)):
                if rlist[k] in letters:
                    rlist[k] = str(a)
        for k in range(0,len(rlist)):
            if type(rlist[k]) == type(sin):
                ag = str(rlist[k])
                rlist[k] = ag[ag.rfind(" ")+1:ag.rfind(">")]
            if rlist[k] == "^":
                rlist[k] = "**"
        return eval("".join(rlist))

class Constraint():
    global SubEntities, constraints
    def __init__(self,nbt_ : dict = {}):
        self.UUID = ["Constraint",0,0,0]
        self.IsRelative, self.Relative, self.Applies = False, [], [None,None]
        self.Angle, self.stableAngle, self.AngleCForce, self.CAngle = [0,0], 0, Formula("a^3"), 0
        self.Length, self.stableLength, self.LengthCForce = [0,0], 0, ""
        self.Width = 3
        self.RenderPriority, self.Sprite, self.SpriteVar = 0, 0, 0
        data_merge(self,nbt_)

    def __getitem__(self,item_):
        return getattr(self,item_)
    
    def __setitem__(self,item_,g):
        setattr(self,item_,g)
    
    def unpack(self,sentity,var_ : dict):
        attributes_ = ["Relative","Applies","Angle","stableAngle","AngleCForce","Width","Length","stableLength","LengthCForce"]
        attributestype = ["refs","refs","formula","formula","cformula","formula","formula","formula","cformula"]
        self.UUID = ["Constraint",random.randint(-2147483648,2147483647),random.randint(-2147483648,2147483647),random.randint(-2147483648,2147483647)]
        self.CAngle *= pi/180
        for k in range(0,len(attributes_)):
            if attributestype[k] == "refs":
                kl = getattr(self,attributes_[k])
                for l in range(0,len(kl)):
                    kl[l] = sentity.SubEntities[kl[l]]
            elif attributestype[k] == "formula":
                kl = getattr(self,attributes_[k])
                if type(kl) == Formula:
                    self[attributes_[k]].calc_for(var_)

class SubEntity():
    global SubEntities
    def __init__(self,nbt_):
        self.UUID = ["SubEntity",random.randint(-2147483648,2147483647),random.randint(-2147483648,2147483647),random.randint(-2147483648,2147483647)]
        self.Name, self.Weight = "", 1
        self.Velocity, self.Position, self.Vectors = [0,0], None, [0,0]
        self.RenderPriority, self.Sprite, self.SpriteVar = 0, 0, 0
        data_merge(self,nbt_)
    
    def __str__(self):
        return "Pos:"+str(self.Position)
    
    def __getitem__(self,item_):
        return getattr(self,item_)
    
    def __setitem__(self,item_,g):
        setattr(self,item_,g)

@numba.jit(cache=True,parallel=False,fastmath=True)
def renderVisualsOriginal(room0,room1,room2,room3,room4,room5,room6,room7,room8,rotationarray,shape,posx,posy,posz,campos,toreturn,maxroom,angleoffsetx,angleoffsety):
    #Roomlayer organisation = (Posx,Posy,Posz,Dimx,Dimy,properties,palettes,isEntityLayer,doesExist)
    for dim1 in range(0,shape[0]):
        for dim2 in range(0,shape[1]):
            r = 0 #Compteur de salle trouvées
            while room8[r] == True:
                sublayer = 0
                while sublayer < 16:
                    xcontact = posx+((campos+room2[r]+sublayer)*(rotationarray[dim1][dim2][0]))-room0[r]
                    ycontact = posy+((campos+room2[r]+sublayer)*(rotationarray[dim1][dim2][1]))-room1[r]
                    if xcontact > 0 and ycontact > 0 and xcontact < room3[r] and ycontact < room4[r]:
                        xcontact2 = floor(xcontact)
                        ycontact2 = floor(ycontact)
                        if room5[r][xcontact2][ycontact2][6] == 0:
                            sublayer += 2
                            continue
                        type_ = room5[r][xcontact2][ycontact2][sublayer//4]
                        if not type_ < 0:
                            distance = floor(((ycontact*sin(angleoffsety))**2+(xcontact*sin(angleoffsetx))**2+((posz+room2[r]+sublayer)*cos(angleoffsetx))**2)**0.5)
                            if room7[r] == True:
                                type_ = type_%10
                                toreturn[dim1][dim2][0] = floor(room6[r][0][type_][distance][0]*(abs(room5[r][xcontact2][ycontact2][0]//10)/10)) + floor(room6[r][1][type_][distance][0]*(abs(room5[r][xcontact2][ycontact2][1]//10)/10)) + floor(room6[r][2][type_][distance][0]*(abs(room5[r][xcontact2][ycontact2][2]//10)/10)) + floor(room6[r][3][type_][distance][0]*(abs(room5[r][xcontact2][ycontact2][3]//10)/10))
                                toreturn[dim1][dim2][1] = floor(room6[r][0][type_][distance][1]*(abs(room5[r][xcontact2][ycontact2][0]//10)/10)) + floor(room6[r][1][type_][distance][1]*(abs(room5[r][xcontact2][ycontact2][1]//10)/10)) + floor(room6[r][2][type_][distance][1]*(abs(room5[r][xcontact2][ycontact2][2]//10)/10)) + floor(room6[r][3][type_][distance][1]*(abs(room5[r][xcontact2][ycontact2][3]//10)/10))
                                toreturn[dim1][dim2][2] = floor(room6[r][0][type_][distance][2]*(abs(room5[r][xcontact2][ycontact2][0]//10)/10)) + floor(room6[r][1][type_][distance][2]*(abs(room5[r][xcontact2][ycontact2][1]//10)/10)) + floor(room6[r][2][type_][distance][2]*(abs(room5[r][xcontact2][ycontact2][2]//10)/10)) + floor(room6[r][3][type_][distance][2]*(abs(room5[r][xcontact2][ycontact2][3]//10)/10))
                                toreturn[dim1][dim2][3] = distance
                            sublayer = 16
                            r = maxroom
                            continue
                    else:
                        sublayer = 16
                        continue
                    sublayer += 2
                r += 1
    return toreturn

@numba.njit(cache=True,parallel=False,fastmath=True)
def renderVisuals(room0,room1,room2,room3,room4,room5,room6,room7,room8,rotationarray,shape,posx,posy,posz,campos,toreturn,maxroom,angleoffsetx,angleoffsety):
    #Roomlayer organisation = (Posx,Posy,Posz,Dimx,Dimy,properties,palettes,isEntityLayer,doesExist)
    for dim1 in range(0,shape[0]):
        for dim2 in range(0,shape[1]):
            r = 0 #Compteur de salle trouvées
            while room8[r] == True:
                sublayer = 0
                xcontact0 = posx+((campos+room2[r])*(rotationarray[dim1][dim2][0]))-room0[r]
                ycontact0 = posy+((campos+room2[r])*(rotationarray[dim1][dim2][1]))-room1[r]
                s = -campos-room2[r]-((room0[r]-posx)/rotationarray[dim1][dim2][0]) # Solution de l'équation où s = profondeur de droite de l'intersection entre droite de vision et côté x du plan
                s2 = -campos-room2[r]-((room1[r]-posx)/rotationarray[dim1][dim2][1])
                while sublayer < 16:
                    xcontact = xcontact0+sublayer*rotationarray[dim1][dim2][0]
                    ycontact = ycontact0+sublayer*rotationarray[dim1][dim2][1]
                    if xcontact > 0 and ycontact > 0 and xcontact < room3[r] and ycontact < room4[r]:
                        xcontact2 = floor(xcontact)
                        ycontact2 = floor(ycontact)
                        if room5[r][xcontact2][ycontact2][6] == 0:
                            sublayer += 2
                            continue
                        type_ = room5[r][xcontact2][ycontact2][sublayer//4]
                        if not type_ < 0:
                            distance = floor(((ycontact*sin(angleoffsety))**2+(xcontact*sin(angleoffsetx))**2+((posz+room2[r]+sublayer)*cos(angleoffsetx))**2)**0.5)
                            if room7[r] == True:
                                type_ = type_%10
                                toreturn[dim1][dim2][0] = floor(room6[r][0][type_][distance][0]*(abs(room5[r][xcontact2][ycontact2][0]//10)/10)) + floor(room6[r][1][type_][distance][0]*(abs(room5[r][xcontact2][ycontact2][1]//10)/10)) + floor(room6[r][2][type_][distance][0]*(abs(room5[r][xcontact2][ycontact2][2]//10)/10)) + floor(room6[r][3][type_][distance][0]*(abs(room5[r][xcontact2][ycontact2][3]//10)/10))
                                toreturn[dim1][dim2][1] = floor(room6[r][0][type_][distance][1]*(abs(room5[r][xcontact2][ycontact2][0]//10)/10)) + floor(room6[r][1][type_][distance][1]*(abs(room5[r][xcontact2][ycontact2][1]//10)/10)) + floor(room6[r][2][type_][distance][1]*(abs(room5[r][xcontact2][ycontact2][2]//10)/10)) + floor(room6[r][3][type_][distance][1]*(abs(room5[r][xcontact2][ycontact2][3]//10)/10))
                                toreturn[dim1][dim2][2] = floor(room6[r][0][type_][distance][2]*(abs(room5[r][xcontact2][ycontact2][0]//10)/10)) + floor(room6[r][1][type_][distance][2]*(abs(room5[r][xcontact2][ycontact2][1]//10)/10)) + floor(room6[r][2][type_][distance][2]*(abs(room5[r][xcontact2][ycontact2][2]//10)/10)) + floor(room6[r][3][type_][distance][2]*(abs(room5[r][xcontact2][ycontact2][3]//10)/10))
                                toreturn[dim1][dim2][3] = distance
                            sublayer = 16
                            r = maxroom
                            continue
                    else:
                        sublayer = 16
                        continue
                    sublayer += 2
                r += 1
    return toreturn

#----------------------------------------------------------------------------------------------
def Physic_Engine():
    global Rooms, Entities
    for t in range(0,len(Rooms)):
        for k in range(0,len(Entities)):
            pass

def launch_interface():
    global windowHeight, windowLength, scrn, mousepos, hold, depth, inputxt, input_on
    scrn = pygame.display.set_mode((0,0))
    running = True
    KKK = 0
    #load_room_visuals("/rooms/SU_A01.txt",24,1920)
    pygame.display.set_caption("nmtory interface")
    scrn = pygame.display.set_mode((windowLength, windowHeight))
    while running == 1:
        pygame.display.flip()
        pressed = tuple(pygame.key.get_pressed())
        #print(pressed)
        #try:
            #print(pressed.index(True))
        #except ValueError:
            #pass
        pressed = tuple(pygame.key.get_pressed())
        mouse_pressed = tuple(pygame.mouse.get_pressed())
        mousepos = list(pygame.mouse.get_pos())

def Script_interpreter(path : str):
    global AEntities
    with open(path.replace("/","\\").replace("here",location)) as k:
        k = k.readlines()
        while True:
            try:
                kword = k[0][:k[0].index(" ")]
            except ValueError:
                kword = k[0].strip()
            if kword == "CreateEntityType":
                a = Entity(ast.literal_eval((k[0][k[0].index(" ")+1:k[0].index("}")+1])))
                k = k[1:]
            #===================================================================#
            #                         Entity Interpret
            #===================================================================#
                while True:
                    if k[0][0] == "\t":
                        #-------------------------- SUBENTITY INTERPRET -------------------------
                        if k[0][1:k[0].index(" ")] == "CreateSubEntity":
                            #Securité pour pas que les gens fassent de la merde avec le eval
                            b = k[0][k[0].index(" ")+1:k[0].index("}")+1]
                            if "Formula(" in b:
                                b = b[:b.index("Formula(")] + "\"\"" + b[b.index(")",b.index("Formula("))+1:]
                                try:
                                    b = SubEntity(ast.literal_eval(b))
                                    b = SubEntity(eval(k[0][k[0].index(" ")+1:k[0].index("}")+1]))
                                except ValueError:
                                    raise SyntaxError
                            else:
                                b = SubEntity(ast.literal_eval(k[0][k[0].index(" ")+1:k[0].index("}")+1]))
                            a.SubEntities[b.Name] = b
                            if "Origin" in k[0]:
                                a.Origin = b
                            k = k[1:]
                        elif k[0][1:k[0].index(" ")] == "CreateConstraint":
                            #Securité pour pas que les gens fassent de la merde avec le eval
                            b = k[0][k[0].index(" ")+1:k[0].index("}")+1]
                            if "Formula(" in b:
                                b = b[:b.index("Formula(")] + "\"\"" + b[b.index(")",b.index("Formula("))+1:]
                                try:
                                    b = Constraint(ast.literal_eval(b))
                                    b = Constraint(eval(k[0][k[0].index(" ")+1:k[0].index("}")+1]))
                                except ValueError:
                                    raise SyntaxError
                            else:
                                b = Constraint(ast.literal_eval(k[0][k[0].index(" ")+1:k[0].index("}")+1]))
                            a.Constraints.append(b)
                            k = k[1:]
                    else:
                        AEntities[a.Name] = a
                        break
            elif kword == "CreateRegion":
                a = Region(ast.literal_eval((k[0][k[0].index(" ")+1:k[0].index("}")+1])))
                k = k[1:]
            elif kword == "End":
                break
            else:
                k = k[1:]

def Summon(entityname,pos : list = [],vars : dict = {}):
    sentity = deepcopy(AEntities[entityname])
    sentity.Origin.Position = [pos[0],pos[1]]
    sentity.Layer = pos[2]
    for k in range(0,len(sentity.Constraints)):
        sentity.Constraints[k].unpack(sentity,vars)
    sentity["Origin"].Position = pos
    #Scripte qui place sur la cart,e les sous entitées générées
    for k in range(0,len(sentity.Constraints)):
        a = sentity.Constraints[k]
        #Cherche à quelle entité déterminer la position
        if a.Applies[0].Position == None:
            b = a.Applies[0],a.Applies[1]
        else:
            b = a.Applies[1],a.Applies[0]
        b[0].Position = b[1].Position[:]
        b[0].Position[0] += round(a.stableLength*sin(a.CAngle),5)
        b[0].Position[1] += round(a.stableLength*cos(a.CAngle),5)
    Entities.append(sentity)

#Load palettes
a = os.listdir(location+"\\Assets\\Palettes")
for l in a:
    with open(location+"\\Assets\\Palettes\\"+str(l)) as k:
        palettes[l[:l.index(".")]] = convert_palette(eval("".join(k.readlines())))

#Script_interpreter("here/Puffercat.txt")
#Summon("player",[8,5,1],{"x":1})