from edvUtils import *
import os
import time
import sys
global themes, toolboxes, activetile, MainEnv, testt, uwuland
themes = {}
camPos = [0,0,-10]
toolboxes = []
activetile = None
TileButtons = []
inputref = None
mousepos = []
clock = pygame.time.Clock()
testt = None
uwuland = False

class edtEnvironment():
    def __init__(self,nbt_):
        self.doRefresh = True
        self.Dimensions, self.Position = [0,0], [0,0]
        self.LayersN = 0
        self.BorderColor = (0,0,0)
        self.subLayers = rnk()
        self.availableColors = [True,numpy.asarray((255,0,0),dtype=numpy.uint8),True,numpy.asarray((0,255,0),dtype=numpy.uint8),True,numpy.asarray((0,0,255),dtype=numpy.uint8),True,numpy.asarray((255,255,0),dtype=numpy.uint8),True,numpy.asarray((255,0,255),dtype=numpy.uint8),True,numpy.asarray((0,255,255),dtype=numpy.uint8)]
        self.materialLayers = dict()
        data_merge(self,nbt_)
        self.__uicons = {"pen":pygame.font.SysFont("Segoe UI Symbol",20),"none":pygame.Surface((0,0)),"scale":pygame.font.SysFont("Segoe UI Symbol",20),"material0":pygame.font.SysFont("Segoe UI Symbol",20),"material1":pygame.font.SysFont("Segoe UI Symbol",35),"material2":pygame.font.SysFont("Segoe UI Symbol",55),"material3":pygame.font.SysFont("Segoe UI Symbol",80)}
        self.__uiconsoffset = {"pen":(0,-10),"none":(0,0),"scale":(-15,-15),"material0":(-15,-15),"material1":(-25,-25),"material2":(-35,-35),"material3":(-60,-60)}
        self.__uicons["pen"] = self.__uicons["pen"].render("",0,(255,255,255))
        self.__uicons["scale"] = self.__uicons["scale"].render("",0,(255,255,255))
        self.__uicons["material0"] = self.__uicons["material0"].render("",0,(255,255,255))
        self.__uicons["material1"] = self.__uicons["material1"].render("",0,(255,255,255))
        self.__uicons["material2"] = self.__uicons["material2"].render("",0,(255,255,255))
        self.__uicons["material3"] = self.__uicons["material3"].render("",0,(255,255,255))
        self.activeuicon = "none"

    def matcursorcolorchange(self,ncolor):
        ncolor = (abs(floor(ncolor[0]-1)),abs(floor(ncolor[1]-1)),abs(floor(ncolor[2]-1)))
        self.__uicons = {"pen":pygame.font.SysFont("Segoe UI Symbol",20),"none":pygame.Surface((0,0)),"scale":pygame.font.SysFont("Segoe UI Symbol",20),"material0":pygame.font.SysFont("Segoe UI Symbol",20),"material1":pygame.font.SysFont("Segoe UI Symbol",35),"material2":pygame.font.SysFont("Segoe UI Symbol",55),"material3":pygame.font.SysFont("Segoe UI Symbol",80)}
        self.__uicons["material0"] = self.__uicons["material0"].render("",1,ncolor,(0,0,0))
        self.__uicons["material1"] = self.__uicons["material1"].render("",1,ncolor,(0,0,0))
        self.__uicons["material2"] = self.__uicons["material2"].render("",1,ncolor,(0,0,0))
        self.__uicons["material3"] = self.__uicons["material3"].render("",1,ncolor,(0,0,0))

    def geticon(self):
        return (self.__uicons[self.activeuicon],self.__uiconsoffset[self.activeuicon])

    def seticon(self,a):
        self.activeuicon = a
        if a == "none":
            pygame.mouse.set_visible(True)
        else:
            pygame.mouse.set_visible(False)

    def __getitem__(self,item_):
        return getattr(self,item_)
    
    def __setitem__(self,item_,g):
        setattr(self,item_,g)
        if item_ == "Layer":
            self.Root.subLayers.changeRnk(self,g)
    
    def addSubLayer(self,nbtn_ : dict):
        nbtn_["Root"] = self
        a = subLayer(nbtn_)
        self.subLayers.add(a,a.subLayer)
        
    def Graphics(self):
        if not type(self) == subLayer:
            if self.doRefresh == True:
                toblit = pygame.Surface((self.Dimensions[0],self.Dimensions[1]))
                for l in self.subLayers.returnOrderedlist(camPos[2]):
                    a = l.returnGraphics()
                    a.set_colorkey((255,255,255))
                    if l.Layer - camPos[2] > 0:
                        a.set_alpha((1/((l.Layer - camPos[2])*5.0))*255+25)
                    elif l.Layer - camPos[2] < 0:
                        a.set_alpha((1/((l.Layer - camPos[2])*-10.0))*255+25)
                    toblit.blit(a,(l.Position[0]-self.Position[0],l.Position[1]-self.Position[1]))
                toblit = pygame.transform.scale(toblit,(toblit.get_size()[0]*ReScaleCoef,toblit.get_size()[1]*ReScaleCoef))
                self.RSurf = toblit.copy()
                self.doRefresh = False
                return toblit
            else:
                return self.RSurf
    
    def returnRotations(self,material):
        p = list()
        for k in self.materialLayers[material].values():
            p.append(k[0].n)
        return p

MainEnv = edtEnvironment({"Dimensions":[windowLength/ReScaleCoef,windowHeight/ReScaleCoef],"Position":[0,0]})

@numba.njit(nopython=True,cache=True,nogil=True)
def fastconvert(p2,p3,toblit,toblit2,shape):
    """Please do not use this under any circumstances. For leditor render purpose only"""
    for dim1 in range(0,shape[1]):
        for dim2 in range(0,shape[2]):
            p3[dim1][dim2][0] = 10
            p2[dim1][dim2][6] = 0
    for layer in range(0,4):
        for dim1 in range(0,shape[1]):
            for dim2 in range(0,shape[2]):
                if toblit[layer][dim1][dim2] == 0:
                    p2[dim1][dim2][layer] = -1 - (p3[dim1][dim2][layer]*10)
                elif toblit[layer][dim1][dim2] == 255:
                    p2[dim1][dim2][layer] = 2 + (p3[dim1][dim2][layer]*10)
                    p2[dim1][dim2][6] = 1
                elif toblit[layer][dim1][dim2] == 65280:
                    p2[dim1][dim2][layer] = 1 + (p3[dim1][dim2][layer]*10)
                    p2[dim1][dim2][6] = 1
                elif toblit[layer][dim1][dim2] == 16711680:
                    p2[dim1][dim2][layer] = 0 + (p3[dim1][dim2][layer]*10)
                    p2[dim1][dim2][6] = 1
                elif toblit[layer][dim1][dim2] == 65535:
                    p2[dim1][dim2][layer] = 3 + (p3[dim1][dim2][layer]*10)
                    p2[dim1][dim2][6] = 1
                elif toblit[layer][dim1][dim2] == 16711935:
                    p2[dim1][dim2][layer] = 4 + (p3[dim1][dim2][layer]*10)
                    p2[dim1][dim2][6] = 1
                elif toblit[layer][dim1][dim2] == 16776960:
                    p2[dim1][dim2][layer] = 5 + (p3[dim1][dim2][layer]*10)
                    p2[dim1][dim2][6] = 1
    for dim1 in range(0,shape[1]):
        for dim2 in range(0,shape[2]):
            p2[dim1][dim2][4] = toblit2[dim1][dim2][0]-128
            p2[dim1][dim2][5] = 10-floor(toblit2[dim1][dim2][1]*(10/255))
    return p2

class subLayer(edtEnvironment):
    def __init__(self,nbt_):
        super().__init__(nbt_)
        self.Layer = 0
        self.subLayer = 0
        self.Name = "input"
        self.PlacedTiles = rnk()
        self.PlacedEffectsRnk = rnk()
        self.PlacedEffectsDict = dict()
        self.PlacedMaterialLayers = dict()
        self.skipTileRender = False
        self.skipEffectRender = False

    def returnGraphics(self):
        if self.doRefresh == True:
            try:
                toblit = pygame.Surface(self.Dimensions, pygame.SRCALPHA)
            except pygame.error as err:
                raiseError(err)
                self.Dimensions = [0,0]
                toblit = pygame.Surface(self.Dimensions, pygame.SRCALPHA)
            except ValueError as err:
                raiseError(err)
                self.Dimensions = [0,0]
                toblit = pygame.Surface(self.Dimensions, pygame.SRCALPHA)
            #Rendu des tuiles posées
            if self.skipTileRender == False:
                toblit2 = pygame.Surface(self.Dimensions, pygame.SRCALPHA)
                for l in self.PlacedTiles.returnOrderedlist():
                    l2 = l.returnGraphics()
                    toblit2.blit(l2,(l.Position[0]-l2.get_size()[0]//2,l.Position[1]-l2.get_size()[1]//2))
                toblit.blit(toblit2,(0,0))
                self.TileRSurf = toblit2.copy()
            else:
                toblit.blit(self.TileRSurf,(0,0))
            #Rendu des matériaux
            for l in self.PlacedMaterialLayers.keys():
                l2 = self.PlacedMaterialLayers[l].returnGraphics()
                l2.set_colorkey((255,255,255))
                toblit.blit(l2,(self.Dimensions[0]//2-l2.get_size()[0]//2,self.Dimensions[1]//2-l2.get_size()[1]//2))
            #Rendu d effects
            if self.skipEffectRender == False:  
                for l in self.PlacedEffectsRnk.returnOrderedlist():
                    toblit.blit(l.returnGraphics(),(0,0))
            #Touche finale
            pygame.draw.rect(toblit,(255,255,254),pygame.Rect(0,0,self.Dimensions[0],self.Dimensions[1]),1)
            self.RSurf = toblit.copy()
            self.skipTileRender = False
            self.skipEffectRender = False
            return toblit
        else:
            return self.RSurf
    
    def openAttrWindow(self):
        global LayerCreationBox, toolboxes, activetile
        LayerCreationBox.Buttons = []
        LayerCreationBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":25,"Theme":themes["Classic"],"Position":[365,8],"Text":"╳","Script":"self.closeSourceWindow(), refreshAll(LayerBox.SandBox)"})
        toolboxes.append(LayerCreationBox)
        LayerCreationBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":23,"Theme":themes["Classic"],"Position":[6,42],"Text":".Dimensions","Script":""})
        LayerCreationBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":23,"Theme":themes["ClassicAlt"],"Position":[185,38],"Text":str(self.Dimensions),"Script":"self.inputText(True,'Dimensions')","StoredReference":self})
        LayerCreationBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":23,"Theme":themes["Classic"],"Position":[6,72],"Text":".Position","Script":""})
        LayerCreationBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":23,"Theme":themes["ClassicAlt"],"Position":[185,68],"Text":str(self.Position),"Script":"self.inputText(True,'Position')","StoredReference":self})
        LayerCreationBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":23,"Theme":themes["Classic"],"Position":[6,102],"Text":".Layer","Script":""})
        LayerCreationBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":23,"Theme":themes["ClassicAlt"],"Position":[185,98],"Text":str(self.Layer),"Script":"self.inputText(True,'Layer')","StoredReference":self})
        LayerCreationBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":23,"Theme":themes["Classic"],"Position":[6,162],"Text":".effects","Script":""})
        LayerCreationBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":23,"Theme":themes["ClassicAlt"],"Position":[185,158],"Text":"","Script":"self.inputText(True,'subLayer')","StoredReference":self})
        LayerCreationBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":23,"Theme":themes["Classic"],"Position":[6,132],"Text":".Name","Script":""})
        LayerCreationBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":23,"Theme":themes["ClassicAlt"],"Position":[185,128],"Text":self.Name,"Script":"self.inputText(True,'Name')","StoredReference":self})

    def placeTile(self,tile,pos = None,sl : int = 0):
        tile = tileRef(tile.returnNBT())
        if pos == None:
            tile["Position"] = [round((mousepos[0]+camPos[0])/ReScaleCoef,0)-self.Position[0],round((mousepos[1]+camPos[1])/ReScaleCoef,0)-self.Position[1]]
        else:
            tile["Position"] = pos
        self.PlacedTiles.add(tile,tile.subLayer)

    def isThereTile(self,distance,pos = None):
        if pos == None:
            pos = mousepos[:]
        pos = [pos[0]/ReScaleCoef-self.Position[0],pos[1]/ReScaleCoef-self.Position[1]]
        for l in self.PlacedTiles:
            dist = ((l.Position[0]-pos[0])**2+(l.Position[1]-pos[1])**2)**0.5
            if dist <= distance:
                return (True,l)
        return (False,None)

    def convert(self): 
        """Returns a usable roomLayer class object of the layer graphics"""
        global testt
        #Fabrique les array2d de la layer
        p = [pygame.Surface(self.Dimensions) for l in range(0,4)]
        props = pygame.Surface(self.Dimensions)
        props.fill((255,255,255))
        for tile in self.PlacedTiles.returnOrderedlist():
            for layer in range(0,len(tile.Reference.RepeatL)):
                for repeatnumber in range(0,tile.Reference.RepeatL[layer]):
                    toblit = tile.returnGraphics(False,layer)
                    pa = sum(tile.Reference.RepeatL[:layer])+tile.subLayer
                    if pa <= 4:
                        p[pa].blit(toblit,(tile.Position[0]-toblit.get_size()[0]//2,tile.Position[1]-toblit.get_size()[1]//2))
            toblitprop = tile.returnGraphics(False,-1)
            props.blit(toblitprop,(tile.Position[0]-toblitprop.get_size()[0]//2,tile.Position[1]-toblitprop.get_size()[1]//2))
        toblit2_ = pygame.surfarray.array3d(props)
        testt = props
        toblit_ = numpy.asarray((pygame.surfarray.array2d(p[0]),pygame.surfarray.array2d(p[1]),pygame.surfarray.array2d(p[2]),pygame.surfarray.array2d(p[3])))
        #Les convertie en format utilisable par le jeu 
        p2_ = numpy.full((self.Dimensions[0],self.Dimensions[1],7),0,dtype=numpy.int8)
        p3_ = numpy.full((self.Dimensions[0],self.Dimensions[1],4),0,dtype=numpy.int8)
        a = fastconvert(p2_,p3_,toblit_,toblit2_,(4,self.Dimensions[0],self.Dimensions[1]))
        return roomLayer({"Position":(self.Position[0],self.Position[1]),"Dimensions":(self.Dimensions[0],self.Dimensions[1]),"Layer":self.Layer,"Properties":a})

    def placeEffect(self,type_,mpos,size,toadd):
        # Script éxécuté dans le cas où le matériau n'existe pas pour la layer en premier lieu
        if not type_.effect in self.PlacedEffectsDict:
            self.PlacedEffectsDict[type_.effect] = effectLayer({"Root":self,"effect":activetile})
            self.PlacedEffectsRnk.add(self.PlacedEffectsDict[type_.effect],0)
        elif not tuple(self.PlacedEffectsDict[type_.effect].effectArray.shape) == tuple(self.Dimensions):
            numpy.reshape(self.PlacedEffectsDict[type_.effect].effectArray,self.Dimensions)
        a = (size**2*2)**0.5
        for dim1 in range(mpos[0]-size,mpos[0]+size):
            for dim2 in range(mpos[1]-size,mpos[1]+size):
                if dim1 >= 0 and dim2 >= 0 and dim1 < self.Dimensions[0] and dim2 < self.Dimensions[1]:
                    if floor(a-((mpos[0]-dim1)**2+(mpos[1]-dim2)**2)**0.5*toadd)*3 > 0:
                        dist = self.PlacedEffectsDict[type_.effect].effectArray[dim1][dim2]+floor(a-((mpos[0]-dim1)**2+(mpos[1]-dim2)**2)**0.5*toadd)*3
                        if dist > 255:
                            dist = 255
                        elif dist < 0:
                            dist = 0
                        self.PlacedEffectsDict[type_.effect].effectArray[dim1][dim2] = dist
        self.PlacedEffectsDict[type_.effect].doRefresh = True

    def placeMaterial(self,pos_,brushsize,isnone : bool = False):
        a = MainEnv.materialLayers[activetile.material][activetile.id][0].n
        if not a in self.PlacedMaterialLayers:   # Au sein de la classe sublayer, la rotation est utilisée pour accéder au materialLayer et non son id
            self.PlacedMaterialLayers[a] = materialLayer({"Root":self,"material":activetile,"rotation":a,"id":activetile.id,"materialtype":activetile.material})
        self.PlacedMaterialLayers[a].placeMaterial(pos_,isnone)
        brushsize *= 4
        for p in range(0,brushsize):
            for pl in range(0,brushsize):
                self.PlacedMaterialLayers[a].placeMaterial((pos_[0]+pl,pos_[1]+p),isnone)
                self.PlacedMaterialLayers[a].placeMaterial((pos_[0]-pl,pos_[1]+p),isnone)
                self.PlacedMaterialLayers[a].placeMaterial((pos_[0]+pl,pos_[1]-p),isnone)
                self.PlacedMaterialLayers[a].placeMaterial((pos_[0]-pl,pos_[1]-p),isnone)
        
class gTheme():
    def __init__(self,nbt_):
        self.BorderColor, self.FontColor, self.Color, self.ColorAlt, self.FontID, self.BorderWidth = (0,0,0), (0,0,0), (0,0,0), (0,0,0), 'Arial', 2
        data_merge(self,nbt_)

    def __getitem__(self,item_):
        return getattr(self,item_)
    
    def __setitem__(self,item_,g):
        setattr(self,item_,g)

class toolBox():
    def __init__(self,nbt_):
        self.Dimensions, self.Position, self.Theme, self.Name = [], [], None, "UwU"
        self.Buttons, self.LocalScroll= [], 0
        self.SandBox = None
        data_merge(self,nbt_)
        self.doRefresh = True
        self.SubFolder = ""
        self.RSurf = pygame.Surface((0,0))

    def __getitem__(self,item_):
        return getattr(self,item_)
    
    def __setitem__(self,item_,g):
        setattr(self,item_,g)
    
    def Graphics(self):
        if self.doRefresh == True:
            rsurf = pygame.Surface(self.Dimensions)
            rsurf.fill(self.Theme.Color)
            pygame.draw.rect(rsurf,self.Theme.BorderColor,pygame.Rect(0,0,self.Dimensions[0],self.Dimensions[1]),self.Theme.BorderWidth)
            #============== Draw Name ====================#
            if type(self) == toolBox and not self.Name == "":
                pygame.draw.rect(rsurf,self.Theme.BorderColor,pygame.Rect(0,0,self.Dimensions[0],40),self.Theme.BorderWidth)
                textf = pygame.font.SysFont("SimSun",34)
                text = textf.render(self.Name,0,self.Theme.FontColor)
                rsurf.blit(text,(self.Theme.BorderWidth*2,self.Theme.BorderWidth*2))
            self.RSurf = rsurf
            #============== Render buttons ===============#
            for l in self.Buttons:
                rsurf.blit(l.returnGraphics(),(l.Position[0],l.Position[1]+self.LocalScroll))
            self.doRefresh = False
            #============== Render sanbox ================#
            l = self.SandBox
            if not l == None:
                rsurf.blit(l.Graphics(),l.Position)
            return rsurf
        else:
            return self.RSurf

    def addButton(self,nbt_i,istilepreview : bool = False):
        nbt_i["Root"] = self
        a = toolButton(nbt_i)
        a.returnGraphics()
        self.Buttons.append(a)
        self.doRefresh = True
        if istilepreview == True:
            TileButtons.append(a)

    def addSandBox(self,nbt_i):
        nbt_i["Root"] = self
        a = internalUI(nbt_i)
        a.Graphics()
        self.SandBox = a
        self.doRefresh = True
    
    def whatsOnMyMouse(self,pos):
        for l in self.Buttons:
            if pos[0] >= l.Position[0] and pos[0] <= l.Position[0]+l.Dimensions[0]:
                if pos[1] >= l.Position[1] and pos[1] <= l.Position[1]+l.Dimensions[1]:
                    return {"button":l,"window":self}
        l = self.SandBox
        if not l == None:
            if pos[0] >= l.Position[0] and pos[0] <= l.Position[0]+l.Dimensions[0]:
                if pos[1] >= l.Position[1] and pos[1] <= l.Position[1]+l.Dimensions[1]:
                    return l.whatsOnMyMouse((pos[0]-l.Position[0],pos[1]-l.Position[1]-l.LocalScroll))
            return {"button":None,"window":self}
        return {"button":None,"window":self}

    def QuickMenuSelect(self,folder):
        """Ouvre un fichier et le converti en interface utilisable ou bien en tuiles utilisables"""
        folder = folder.replace("here",location).replace("selfname",self.SubFolder+self.Name)
        a = os.listdir(folder)
        #====================== Ouverture d'un répertoire =========================
        self.Buttons = []
        k = -1
        p = -1
        while True:
            k += 1
            p += 1
            if p >= len(a):
                break
            kl = a[p]
            if not kl == "Materials" and not kl == "Effects":
                self.addButton({"SubFolder":"","Name":kl,"Text":kl,"FontSize":15,"AffectedByScrolling":False,"Position":[5,5+k*20],"Theme":themes["QuickMenu"],"Script":f'self.QuickMenuSelectOption("here\Assets\Editor\Placeables\{kl}")'})
                self.addButton({"SubFolder":"","Name":kl,"Text":"","FontSize":17,"AffectedByScrolling":False,"Position":[110,5+k*20],"Theme":themes["QuickMenu"],"Script":f'self.QuickMenuSelectOption("here\Assets\Editor\Placeables\{kl}")'})
                self.doRefresh = True
            else:
                k -= 1

    def QuickMenuSelect2(self,folder):
        """Ouvre un fichier et le converti en interface utilisable ou bien en tuiles utilisables"""
        folder = folder.replace("here",location).replace("selfname",self.SubFolder+self.Name)
        a = os.listdir(folder)
        #====================== Ouverture d'un répertoire =========================
        self.Buttons = []
        k = -1
        p = -1
        while True:
            k += 1
            p += 1
            if p >= len(a):
                break
            kl = a[p]
            if not kl == "!None.edtmat" and not kl == "!SetMaterials.txt":
                self.addButton({"SubFolder":"","Name":kl,"Text":kl[:kl.index(".")],"FontSize":17,"AffectedByScrolling":False,"Position":[5,5+k*20],"Theme":themes["QuickMenu2"],"Script":f'self.QuickMenuSelectOption2("{kl[:kl.index(".")]}")'})
                self.addButton({"SubFolder":"","Name":kl,"Text":"","FontSize":17,"AffectedByScrolling":False,"Position":[110,5+k*20],"Theme":themes["QuickMenu2"],"Script":f'self.QuickMenuSelectOption2("{kl[:kl.index(".")]}")'})
                self.doRefresh = True
            else:
                k -= 1

class internalUI(toolBox):
    def __init__(self,nbt_):
        super().__init__(nbt_)
        data_merge(self,nbt_)
    
    def clearButtons(self):
        self.Buttons = []
        self.doRefresh = True
    
    def scrollTile(self,y):
        for i in range(0,len(self.Buttons)):
            if self.Buttons[i].Tile == activetile.Reference:
                if i-y >= len(self.Buttons):
                    self.Buttons[1].selectTile()
                    return
                elif self.Buttons[i-y].Tile == None:
                    self.Buttons[-1].selectTile()
                    return
                self.Buttons[i-y].selectTile()
                return

class toolButton():
    def __init__(self,nbt_):
        self.Dimensions, self.Position, self.Theme, self.Text = [], [], None, "=w="
        self.Sprite, self.GraphicType = None, "Unicode"
        self.doRefresh, self.Tile = True, None
        data_merge(self,nbt_)

    def __getitem__(self,item_):
        return getattr(self,item_)
    
    def __setitem__(self,item_,g):
        setattr(self,item_,g)

    def selectTile(self,isnone=False):
        global activetile
        if isinstance(activetile,tileRef) and isnone == False:
            prerot = activetile.Rotation
        else:
            prerot = 0
        if isnone == True:
            activetile = None
            for k in range(0,len(TileButtons)):
                TileButtons[k].Theme = themes["SpritePreview"]
            self.Theme = themes["SpritePreviewSelect"]
            self.Root.doRefresh = True
            self.Root.Root.doRefresh = True
            return
        for k in range(0,len(TileButtons)):
            TileButtons[k].Theme = themes["SpritePreview"]
        self.Theme = themes["SpritePreviewSelect"]
        activetile = tileRef({"Reference":self.Tile,"Rotation":prerot,"ScaleCoef":[1,1]})
        self.Root.doRefresh = True
        self.Root.Root.doRefresh = True

    def returnGraphics(self):
        """Retourne les graphiques de l'objet sous forme de pygame.Surface"""
        if self.doRefresh == True:
            if self.GraphicType == "Unicode":
                textf = pygame.font.SysFont(self.Theme.ButtonFont,self.FontSize)
                b = self.Text
                if "onRender:" in b:
                    b = eval(self.Text.replace("onRender:",""))
                textf = textf.render(b,0,self.Theme.FontColor)
                dimensions = textf.get_size()
                if not self.Theme.ButtonBorderWidth ==0:
                    pygame.draw.rect(textf,self.Theme.BorderColor,pygame.Rect(0,0,dimensions[0],dimensions[1]),self.Theme.ButtonBorderWidth)
                self.Dimensions = dimensions
                self.SurfSave = textf
                return textf
            #Graphic si le bouton est un preview de tile
            elif self.GraphicType == "spritePreview":
                returnsurf = pygame.Surface((self.Dimensions[0],self.Dimensions[1]))
                dimensions = self.Dimensions
                if not self.Theme.ButtonBorderWidth ==0:
                    pygame.draw.rect(returnsurf,self.Theme.BorderColor,pygame.Rect(0,0,dimensions[0],dimensions[1]),self.Theme.ButtonBorderWidth)
                #Rescale preview
                dimensions = (self.Dimensions[0]-self.Theme.ButtonBorderWidth*4,self.Dimensions[1]-self.Theme.ButtonBorderWidth*4)
                toblit = self.Tile.Preview
                scalecoef = dimensions[0]/max(toblit.get_size()[0],toblit.get_size()[1])
                toblit = pygame.transform.scale(toblit,(toblit.get_size()[0]*scalecoef,toblit.get_size()[1]*scalecoef))
                toblit.set_colorkey((255,255,255))
                returnsurf.blit(toblit,(dimensions[0]-toblit.get_size()[0]+self.Theme.ButtonBorderWidth*2,dimensions[1]-toblit.get_size()[1]+self.Theme.ButtonBorderWidth*2))
                return returnsurf
        else:
            return self.SurfSave

    def execScript(self):
        if not self.Script == "":
            if "Reference." in self.Script:
                b = "MainEnv.subLayers["+str(MainEnv.subLayers.getIndex(self.Reference)-MainEnv.subLayers.len()+1)+"]"
                try:
                    eval(self.Script.replace("Reference",b))
                except BaseException:
                    raise SyntaxError("Invalid script "+self.Script.replace("Reference",b)+" executed")
                return
            try:
                eval(self.Script)
            except BaseException as err:
                raise SyntaxError("Invalid script "+self.Script+" executed, caused the error: "+str(err)+", The script was executed by object "+str(self)+" which belongs to object "+str(self.Root))
    
    def closeSourceWindow(self):
        toolboxes.pop(toolboxes.index(getattr(self,"Root")))
    
    def openFolder(self,folder,type_ : str = "tilebox"):
        """Ouvre un fichier et le converti en interface utilisable ou bien en tuiles utilisables"""
        folder = folder.replace("here",location).replace("selfname",self.SubFolder+self.Name)
        a = os.listdir(folder)
        #====================== Ouverture d'un répertoire =========================
        #Si l'ouverture se produit depuis une toolbox directement
        if type(self.Root) == toolBox:
            self.Root.SandBox.clearButtons()
            for k in range(0,len(a)):
                kl = a[k]
                if not "." in kl and type_ == "tilebox":
                    self.Root.SandBox.addButton({"SubFolder":"","Name":kl,"Text":" "+kl,"FontSize":25,"AffectedByScrolling":True,"Position":[5,10+k*40],"Theme":themes["ClassicAlt"],"Script":r'self.openFolder("here\Assets\Editor\Placeables\selfname")'})
                    self.Root.doRefresh = True
                    self.Root.SandBox.LocalScroll = 0
        #Script diffère légèrement en fonction de où est executé la 'fonction', si elle est executé depuis la sandbox, elle doit remonter à la Root
        elif type(self.Root) == internalUI:
            self.Root.clearButtons()
            self.Root.Root.doRefresh = True
            n = 0
            for k in range(0,len(a)):
                kl = a[k]
                if not "." in kl and type_ == "tilebox":
                    self.Root.addButton({"SubFolder":self.Name+"\\","Name":kl,"Text":" "+kl,"FontSize":25,"AffectedByScrolling":True,"Position":[5,10+k*40],"Theme":themes["ClassicAlt"],"Script":r'self.openFolder("here\Assets\Editor\Placeables\selfname")'})
                    self.Root.Root.doRefresh = True
                    self.Root.LocalScroll = 0
        #====================== Ouverture d'un dossier de tuile ============================================
                #Open répertoire de tiles en détéctant si un fichier nommé SetProperties.txt est présent
                if "!SetProperties.txt" in a and ".png" in kl and not ".txt" in kl:
                    if kl == "!None.png":
                        self.Root.addButton({"SubFolder":self.Name+"\\","GraphicType":"Unicode","FontSize":49,"Name":"","Text":"","AffectedByScrolling":True,"Position":[10+n%3*80,10+n//3*80],"Theme":themes["SpritePreviewSelect"],"Script":r'self.selectTile(True)',"Tile":None,"Dimensions":[65,65]},True)
                        n += 1
                        self.Root.Root.doRefresh = True
                        self.Root.LocalScroll = 0
                    else:
                        with open(folder+"\\!SetProperties.txt") as properties:
                            properties = ast.literal_eval(properties.readlines()[0])
                        #self.Root.addButton({"SubFolder":self.Name+"\\","AffectedByScrolling":True,"Position":[5+k%3*40,10+k//3*40],"Theme":themes["ClassicAlt"],"Script":r'self.openFolder("here\Assets\Editor\Placeables\selfname")',"GraphicType":"tilePreview","Tile":convert_tile(folder+"\\"+kl)})
                        self.Root.addButton({"SubFolder":self.Name+"\\","Name":kl,"Text":" "+kl,"FontSize":25,"AffectedByScrolling":True,"Position":[10+n%3*80,10+n//3*80],"Theme":themes["SpritePreview"],"Script":r'self.selectTile()',"GraphicType":"spritePreview","Tile":convert_tile(folder+"\\"+kl),"Dimensions":[65,65]},True)
                        n += 1
                        self.Root.Root.doRefresh = True
                        self.Root.LocalScroll = 0
        #======================= Ouverture d'un dossier d'effects' ========================================
                elif "!SetEffects.txt" in a and ".edteff" in kl and not ".txt" in kl:
                    if kl == "!None.png":
                        self.Root.addButton({"SubFolder":self.Name+"\\","GraphicType":"Unicode","FontSize":23,"Name":"","Text":" x ","AffectedByScrolling":True,"Position":[5,10],"Theme":themes["SpritePreviewSelect"],"Script":r'self.selectTile(True)',"Tile":None,"Dimensions":[65,65]},True)
                        n += 1
                        self.Root.Root.doRefresh = True
                        self.Root.LocalScroll = 0
                    else:
                        self.Root.addButton({"SubFolder":self.Name+"\\","Name":kl,"Text":" "+kl+" ","FontSize":23,"AffectedByScrolling":True,"Position":[5,10+n*35],"Theme":themes["SpritePreview"],"Script":f'self.selectEffect("{kl[:kl.index(".")]}")',"GraphicType":"Unicode","Dimensions":[65,65]},True)
                        n += 1
                        self.Root.Root.doRefresh = True
                        self.Root.LocalScroll = 0
        #====================== Ouverture d'un dossier de matériau ========================================
                elif "!SetMaterials.txt" in a and ".edtmat" in kl and not ".txt" in kl:
                    if kl == "!None.edtmat":
                        self.Root.addButton({"SubFolder":self.Name+"\\","GraphicType":"Unicode","FontSize":23,"Name":"","Text":" "+kl+" ","AffectedByScrolling":True,"Position":[5,7],"Theme":themes["ClassicAlt"],"Script":r'self.selectMaterial(True)',"Tile":None,"Dimensions":[65,65]},True)
                        n += 1
                        self.Root.Root.doRefresh = True
                        self.Root.LocalScroll = 0
                    else:
                        self.Root.addButton({"SubFolder":self.Name+"\\","Name":kl,"Text":" "+kl+" ","FontSize":23,"AffectedByScrolling":True,"Position":[5,7+n*35],"Theme":themes["ClassicAlt"],"Script":"","GraphicType":"Unicode","Dimensions":[65,65]},True)
                        for j in range(0,7):
                            if kl[:kl.index(".")] in MainEnv.materialLayers and j in MainEnv.materialLayers[kl[:kl.index(".")]]:
                                self.Root.addButton({"SubFolder":self.Name+"\\","Name":kl,"Text":"","FontSize":23,"AffectedByScrolling":True,"Position":[5+j*35,7+n*35+35],"Theme":themes["ClassicAlt"],"Script":f'self.selectMaterial("{kl[:kl.index(".")]}",{str(j)}), self.addMaterialLayer("{kl[:kl.index(".")]}",{str(j)})',"GraphicType":"Unicode","Dimensions":[65,65],"Reference":self},True)
                            else:
                                self.Root.addButton({"SubFolder":self.Name+"\\","Name":kl,"Text":"","FontSize":23,"AffectedByScrolling":True,"Position":[5+j*35,7+n*35+35],"Theme":themes["ClassicAlt"],"Script":f'self.selectMaterial("{kl[:kl.index(".")]}",{str(j)}), self.addMaterialLayer("{kl[:kl.index(".")]}",{str(j)})',"GraphicType":"Unicode","Dimensions":[65,65],"Reference":self},True)
                        for j in range(0,7):
                            if kl[:kl.index(".")] in MainEnv.materialLayers and j in MainEnv.materialLayers[kl[:kl.index(".")]]:
                                self.Root.addButton({"SubFolder":"g","Name":"p","Text":str(MainEnv.materialLayers[kl[:kl.index(".")]][j][0].n),"FontSize":14,"AffectedByScrolling":True,"Position":[12+j*35,23+n*35+35],"Theme":themes["Error"],"Script":f'self.selectMaterial("{kl[:kl.index(".")]}",{str(j)}), self.addMaterialLayer("{kl[:kl.index(".")]}",{str(j)})',"GraphicType":"Unicode","Dimensions":[65,65],"Reference":self},True)
                            else:
                                pass
                        n += 2
                        self.Root.Root.doRefresh = True
                        self.Root.LocalScroll = 0

    def addLayer(self):
        global activetile
        try:
            last = self.Root.SandBox.Buttons[-1]
            cpos = self.Root.SandBox.Buttons[-1].Position[1] + 30
        except IndexError:
            last = None
            cpos = 5
        if not last == None and last.Name == "Instruction":
            return
        MainEnv.addSubLayer({"Dimensions":[0,0],"Position":[0,0]})
        self.Root.SandBox.addButton({"SubFolder":"","Name":"Instruction","GraphicType":"Unicode","FontSize":23,"Theme":themes["ClassicAlt"],"Position":[195,cpos],"Text":"","Script":"Reference.openAttrWindow()","Reference":MainEnv.subLayers.returnLastAdded()})
        self.Root.SandBox.addButton({"SubFolder":"","Name":"Instruction","GraphicType":"Unicode","FontSize":23,"Theme":themes["ClassicAlt"],"Position":[220,cpos],"Text":"","Script":"self.trashLayer()","Reference":MainEnv.subLayers.returnLastAdded()})
        self.Root.SandBox.addButton({"SubFolder":"","Name":"Instruction","GraphicType":"Unicode","FontSize":20,"Theme":themes["ClassicAlt"],"Position":[5,cpos],"Text":" Select Area","Script":"self.selectLayer()","Reference":MainEnv.subLayers.returnLastAdded()})
        self.Root.SandBox.doRefresh = True
        self.Root.doRefresh = True
        activetile = [MainEnv.subLayers.returnLastAdded(),self.Root,"layercreation",False]
        MainEnv.seticon("scale")

    def selectLayer(self):
        global camPos, MainEnv
        camPos[2] = self.Reference.Layer
        MainEnv.doRefresh = True

    def inputText(self,ref : bool = False,attribute : str = "Text"):
        self.Theme = themes["InputStandBy"]
        if ref == False:
            ref = self
        else:
            ref = self["StoredReference"]
        global inputref
        inputref = ["",ref,attribute,self,"input",False]
        try:
            inputref[1].Root.doRefresh = True
        except AttributeError:
            inputref[1].doRefresh = True

    def trashLayer(self):
        a = self.Root.Buttons.index(self)
        b = self.Root.Buttons
        MainEnv.subLayers.remove(b[a].Reference)
        MainEnv.doRefresh = True
        for l in range(0,3):
            del b[a-1]
        for k in self.Root.Buttons:
            if b.index(k) >= a-1:
                k.Position[1] -= 30
        self.Root.doRefresh = True
        self.Root.Root.doRefresh = True

    def refresh(self):
        self.doRefresh = True
        self.Root.doRefresh = True
        try:
            self.Root.Root.doRefresh = True
        except AttributeError:
            pass

    def selectEffect(self,type_):
        global activetile
        if type_ == "!None":
            activetile = None
            MainEnv.activeuicon = "none"
            pygame.mouse.set_visible(True)
            return
        color = {"Plates":(200,200,200)}[type_]
        activetile = effect(type_,color)
        MainEnv.seticon("pen")

    def selectMaterial(self,type_,id : int = 0):
        global activetile
        color = {"Plates":(255,0,0),"Concrete":(50,100,255)}[type_]
        if type_ in MainEnv.materialLayers and id in MainEnv.materialLayers[type_]:
            activetile = material(type_,color,MainEnv.materialLayers[type_][id][0].n,id)
        else:
            activetile = material(type_,color,0,id)
        if MaterialsLQuickMenu in toolboxes:
            toolboxes.pop(toolboxes.index(MaterialsLQuickMenu))
        if MaterialsQuickMenu in toolboxes:
            toolboxes.pop(toolboxes.index(MaterialsQuickMenu))
        pygame.mouse.set_visible(False)
        MainEnv.activeuicon = "material0"

    def addMaterialLayer(self,type_,id,mode : bool = False):
        global activetile
        if mode == False:
            if not type_ in MainEnv.materialLayers:
                MainEnv.materialLayers[type_] = dict()
            if not id in MainEnv.materialLayers[type_]:
                for p in range(0,len(self.Root.Buttons)):
                    if type_ in self.Root.Buttons[p].Text:
                        a = self.Root.Buttons[p+1+id]
                MainEnv.materialLayers[type_][id] = (rot(0),a)
                a.Text = ""
                self.Root.doRefresh = True
                self.Root.Root.doRefresh = True
        else:
            if not type_ in MainEnv.materialLayers:
                MainEnv.materialLayers[type_] = dict()
            if not id in MainEnv.materialLayers[type_]:
                for p in range(0,len(Tilebox.SandBox.Buttons)):
                    if type_ in Tilebox.SandBox.Buttons[p].Text:
                        a = Tilebox.SandBox.Buttons[p+1+id]
                        b = MaterialsLQuickMenu.Buttons[id]
                MainEnv.materialLayers[type_][id] = (rot(0),a)
                a.Text = ""
                b.Text = ""
                MaterialsLQuickMenu.doRefresh = True
                Tilebox.doRefresh = True
                Tilebox.SandBox.doRefresh = True
        DefRotationPls.Buttons = list()
        DefRotationPls.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":25,"Theme":themes["Classic"],"Position":[170,8],"Text":"╳","Script":"self.closeSourceWindow(), refreshAll(LayerBox.SandBox)"})
        toolboxes.append(DefRotationPls)
        DefRotationPls.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":13,"Theme":themes["ClassicAlt"],"Position":[4,37],"Text":"Rotation is in degrees,","Script":""})
        DefRotationPls.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":22,"Theme":themes["ClassicAlt"],"Position":[160,67],"Text":"","Script":f'self.delMaterialLayer("{type_}",{id})'})
        DefRotationPls.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":13,"Theme":themes["ClassicAlt"],"Position":[4,52],"Text":"floats or negatives unsupported","Script":""})
        DefRotationPls.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":22,"Theme":themes["ClassicAlt"],"Position":[4,67],"Text":"Rotation:","Script":""})
        DefRotationPls.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":22,"Theme":themes["ClassicAlt"],"Position":[100,67],"Text":str(MainEnv.materialLayers[type_][id][0].n),"Script":"self.inputText(True,'n')","StoredReference":MainEnv.materialLayers[type_][id][0]})

    def QuickMenuSelectOption(self,folder):
        Tilebox.SandBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":23,"Theme":themes["Classic"],"Position":[5,45],"Text":"RoomSize:","Script":""})
        Tilebox.SandBox.Buttons[0].openFolder(folder)
        Tilebox.SandBox.Buttons[-1].selectTile(self)
        if QuickMenu in toolboxes:
          toolboxes.pop(toolboxes.index(QuickMenu))

    def QuickMenuSelectOption2(self,folder):
        if not MaterialsLQuickMenu in toolboxes:
            MaterialsLQuickMenu.Position = [MaterialsQuickMenu.Position[0]+MaterialsQuickMenu.Dimensions[0],MaterialsQuickMenu.Position[1]]
            MaterialsLQuickMenu.Buttons = list()
            toolboxes.append(MaterialsLQuickMenu)
            for j in range(0,7):
                if folder in MainEnv.materialLayers and j in MainEnv.materialLayers[folder]:
                    MaterialsLQuickMenu.addButton({"SubFolder":self.Name+"\\","Name":"p","Text":"","FontSize":23,"AffectedByScrolling":True,"Position":[3,7+j*35],"Theme":themes["ClassicAlt"],"Script":f'Tilebox.SandBox.Buttons[0].openFolder("here\Assets\Editor\Placeables\Materials"), self.selectMaterial("{folder}",{str(j)})',"GraphicType":"Unicode","Dimensions":[65,65],"Reference":self},True)
                    MaterialsLQuickMenu.addButton({"SubFolder":self.Name+"\\","Name":"p","Text":str(MainEnv.materialLayers[folder][j][0].n),"FontSize":14,"AffectedByScrolling":True,"Position":[12,23+j*35],"Theme":themes["Error"],"Script":f'Tilebox.SandBox.Buttons[0].openFolder("here\Assets\Editor\Placeables\Materials"), self.selectMaterial("{folder}",{str(j)})',"GraphicType":"Unicode","Dimensions":[65,65],"Reference":self},True)
                else:
                    MaterialsLQuickMenu.addButton({"SubFolder":self.Name+"\\","Name":"p","Text":"","FontSize":23,"AffectedByScrolling":True,"Position":[3,7+j*35],"Theme":themes["ClassicAlt"],"Script":f'Tilebox.SandBox.Buttons[0].openFolder("here\Assets\Editor\Placeables\Materials"), self.selectMaterial("{folder}",{str(j)}), self.addMaterialLayer("{folder}",{str(j)},True)',"GraphicType":"Unicode","Dimensions":[65,65],"Reference":self},True)

    def delMaterialLayer(self,type_,id):
        global activetile
        del MainEnv.materialLayers[type_][id]
        toolboxes.pop(toolboxes.index(DefRotationPls))
        Tilebox.Buttons[-1].openFolder(r'here\Assets\Editor\Placeables')
        pygame.mouse.set_visible(True)
        activetile = None
        MainEnv.activeuicon = "none"

class tile():
    def __init__(self,nbt_):
        self.Width, self.Height, self.RepeatL = 0, 0, []
        data_merge(self,nbt_)

    def __getitem__(self,item_):
        return getattr(self,item_)
    
    def __setitem__(self,item_,g):
        setattr(self,item_,g)

    def returnGraphicsPreview(self):
        a = self.Preview
        return a.copy()

    def returnGraphicsLayer(self,i : int):
        if i >= 0:
            return self.Layers[i]
        else:
            return self.Collisions

class tileRef():
    def __init__(self,nbt_ : str = {"Rotation":0,"ScaleCoef":[0,0]}):
        self.Rotation, self.ScaleCoef, self.subLayer = 0, [0,0], 0
        data_merge(self,nbt_)

    def __getitem__(self,item_):
        return getattr(self,item_)
    
    def __setitem__(self,item_,g):
        setattr(self,item_,g)
    
    def returnGraphics(self,ispreview : bool = True,i : int = 0):
        if ispreview == True:
            tomodify = self.Reference.returnGraphicsPreview()
        else:
            tomodify = self.Reference.returnGraphicsLayer(i)
        newsurf = pygame.Surface((tomodify.get_size()[0]*2,tomodify.get_size()[1]*2))
        newsurf.fill((255,255,255))
        newsurf.blit(tomodify,(tomodify.get_size()[0]//2,tomodify.get_size()[1]//2))
        tomodify = pygame.transform.rotate(newsurf,self.Rotation)
        tomodify = pygame.transform.scale(tomodify,(self.ScaleCoef[0]*tomodify.get_size()[0],self.ScaleCoef[1]*tomodify.get_size()[1]))
        if not self.subLayer == 0:
            newsurf = pygame.Surface(tomodify.get_size())
            newsurf.fill((255,255,255))
            newsurf.set_alpha(self.subLayer*75)
            tomodify.blit(newsurf,(0,0))
        tomodify.set_colorkey((255,255,255))
        return tomodify

    def returnNBT(self):
        return {"Rotation":self.Rotation,"ScaleCoef":self.ScaleCoef[:],"Reference":self.Reference,"subLayer":self.subLayer}
    
    def getRelativePosition(self,offset : list = [0,0],addrot : int = 0):
        rotation = self.Rotation + addrot
        return [self.Position[0]-offset[1]*sin(rotation/180*pi)-offset[0]*cos(rotation/180*pi),self.Position[1]+offset[0]*sin(rotation/180*pi)+offset[1]*cos(rotation/180*pi)]

class effectLayer():
    def __init__(self,nbt_):
        self.effect = None
        data_merge(self,nbt_)
        self.effectArray = numpy.full((self.Root.Dimensions[0],self.Root.Dimensions[1]),0,dtype=numpy.uint8)
        self.doRefresh = True
    
    def returnGraphics(self):
        a = numpy.full((self.effectArray.shape[0],self.effectArray.shape[1],3),self.effect.color,dtype=numpy.uint8)
        a = pygame.surfarray.make_surface(a).convert_alpha()
        b = pygame.surfarray.pixels_alpha(a)
        b[:,:] = self.effectArray[:,:]
        return a

    def __getitem__(self,item_):
        return getattr(self,item_)
    
    def __setitem__(self,item_,g):
        setattr(self,item_,g)

class effect():
    def __init__(self,material_,color_):
        self.effect = material_
        self.color = color_

class rot():
    def __init__(self,n : int):
        self.n = n

    def __getitem__(self,item_):
        return getattr(self,item_)
    
    def __setitem__(self,item_,g):
        setattr(self,item_,g)

class materialLayer():
    def __init__(self,nbt_):
        self.materialArray = numpy.full((0,0),-1,dtype=numpy.int8)
        self.material = None
        self.rotation = 0
        data_merge(self,nbt_)
        self.rotation = self.rotation*pi/180
        refpos = self.Root.Dimensions
        self.materialArray = numpy.full((refpos[0]//3,refpos[1]//3),-1,dtype=numpy.int8)
        self.dimensions = (refpos[0]*2,refpos[1]*2)
        self.graphics = pygame.Surface(self.dimensions)
        self.graphics.fill((255,255,255))
        #Del les layers inutilisées (inaccessibles)
        a = MainEnv.returnRotations(self.materialtype)
        for p in self.Root.PlacedMaterialLayers:
            if not p in a:
                del self.Root.PlacedMaterialLayers[p]
                break

    def __getitem__(self,item_):
        return getattr(self,item_)
    
    def __setitem__(self,item_,g):
        setattr(self,item_,g)
    
    def placeMaterial(self,pos,isdel : bool = False):
        self.Root.skipTileRender = True
        self.Root.skipEffectRender = True
        pos = [pos[0]-self.Root.Position[0]-self.Root.Dimensions[0]//2,pos[1]-self.Root.Position[1]-self.Root.Dimensions[1]//2]
        pos = [floor(cos(self.rotation)*pos[0]-sin(self.rotation)*pos[1]),floor(cos(self.rotation)*pos[1]+sin(self.rotation)*pos[0])]
        pos = [pos[0]+self.Root.Dimensions[0],pos[1]+self.Root.Dimensions[1]]
        if not (pos[0] >= 0 and pos[1] >= 0 and pos[0] < self.dimensions[0]-1 and pos[1] < self.dimensions[1]-1):
            return False
        if isdel == False:
            self.materialArray[pos[0]//6][pos[1]//6] = activetile.var
            pygame.draw.rect(self.graphics,(self.material.color[0]//4*(activetile.var+1),self.material.color[1]//4*(activetile.var+1),self.material.color[2]//4*(activetile.var+1)),pygame.Rect(pos[0]-(pos[0]%6),pos[1]-(pos[1]%6),6,6),1)
        if isdel == True:
            self.materialArray[pos[0]//6][pos[1]//6] = -1
            pygame.draw.rect(self.graphics,(0,0,0),pygame.Rect(pos[0]-(pos[0]%6),pos[1]-(pos[1]%6),6,6),1)
        self.Root.doRefresh = True
        MainEnv.doRefresh = True
    
    def returnGraphics(self):
        a = self.graphics.copy()
        a = pygame.transform.rotate(a,self.rotation*180/pi)
        return a

    def renderGraphics(self):
        pass

class material():
    def __init__(self,material_,color_,rotation=0,id_=0):
        self.material = material_
        self.color = color_
        self.rotation = rotation
        self.id = id_
        self.var = 0
        self.brushsize = 0

def convert_tile(folder):
    """Convert path to tile image into usable tile by level editor"""
    #Lignes préparatoires
    image = pygame.image.load(folder).convert()
    with open(folder+".txt") as k:
        k = ast.literal_eval(k.readlines()[0])
    layern = len(k["RepeatL"])
    #Commence la conversion : récupère le preview
    k["Preview"] = pygame.Surface((k["Width"],k["Height"]))
    k["Preview"].blit(image,(0,0),pygame.Rect(0,0,k["Width"],k["Height"]))
    k["Layers"] = []
    #Décompose le tile en ses différentes layers
    for lj in range(1,layern+1):
        j = pygame.Surface((k["Width"],k["Height"]))
        j.blit(image,(0,0),pygame.Rect(0,lj*k["Height"],k["Width"],lj*k["Height"]+k["Height"]))
        j.set_colorkey((255,255,255))
        k["Layers"].append(j)
    lj += 1
    j = pygame.Surface((k["Width"],k["Height"]))
    j.blit(image,(0,0),pygame.Rect(0,lj*k["Height"],k["Width"],lj*k["Height"]+k["Height"]))
    k["Collisions"] = j
    if not "OverlapCoef" in k:
        k["OverlapCoef"] = 0.95
    return tile(k)

def launchEditor():
    global windowHeight, windowLength, scrn, mousepos, inputref, MainEnv, activetile, toolboxes, themes, clock, camPos, uwuland, testt
    scrn = pygame.display.set_mode((0,0))
    running = True
    kkk = 0
    #load_room_visuals("/rooms/SU_A01.txt",24,1920-)
    pygame.display.set_caption("nmtory interface")
    scrn = pygame.display.set_mode((windowLength, windowHeight))
    cpos = None
    while running == 1:
        pygame.display.flip()
        pressed = tuple(pygame.key.get_pressed())
        #print(pressed)
        #try:
            #print(pressed.index(True))
        #except ValueError:
            #pass
        #clock.tick()
        #print(clock.get_fps())
        pressed = tuple(pygame.key.get_pressed())
        mouse_pressed = tuple(pygame.mouse.get_pressed())
        mousepos = list(pygame.mouse.get_pos())
        mouseposcopy = mousepos[:]
        #================== LeftClick on toolBox ==========================
        a = whatsOnMyMouse(mousepos)
        mousecoordinates = ((mousepos[0]+camPos[0])//ReScaleCoef, (mousepos[1]+camPos[1])//ReScaleCoef)
        if type(activetile) == tileRef and not mouse_pressed[1] == True:
            if not cpos == None:
                mousepos = cpos[:]
        for event in pygame.event.get():
            #==================================== Se déplacer dans un edtenv ============================================
            if mouse_pressed[1] == True:
                testt = None
                camPos[0] = camPos[0]-(mousepos[0]-cpos2[0])
                camPos[1] = camPos[1]-(mousepos[1]-cpos2[1])
                cpos2 = mouseposcopy[:]
            else:
                cpos2 = mouseposcopy[:]
            #============================= Tout ce qui est relaté aux tuiles ====================================
            if type(activetile) == tileRef:
                #============================= Placer une tuile ====================================
                if type(a["window"]) == subLayer:
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not mouse_pressed[2] == True:
                        a["window"].placeTile(activetile)
                        a["window"].doRefresh = True
                        a["window"].Root.doRefresh = True
                    elif mouse_pressed[2] == True:
                        if mouse_pressed[0] == True:
                            try:
                                a["window"].PlacedTiles.remove(a["window"].isThereTile(5,(mouseposcopy[0]+camPos[0],mouseposcopy[1]+camPos[1]))[1])
                                a["window"].doRefresh = True
                                a["window"].Root.doRefresh = True
                            except ValueError:
                                a["window"].doRefresh = True
                                a["window"].Root.doRefresh = True
                        elif event.type == pygame.MOUSEWHEEL:
                            activetile.subLayer = (activetile.subLayer + event.y)%4
                    #Placer des tuiles adjacentes
                    elif mouse_pressed[0] == True and not a["window"].PlacedTiles == []:
                        for lll in range(0,2):
                            for kj in range(0,2):
                                try:
                                    bl = a["window"].PlacedTiles.returnLastAdded().getRelativePosition([activetile.Reference.Width*abs(lll-1)*activetile.Reference.OverlapCoef+activetile.Reference.Height*lll*activetile.Reference.OverlapCoef,0],180*kj+lll*90)
                                    dist = round(((bl[0]-((mousepos[0]+camPos[0])/ReScaleCoef-a["window"].Position[0]))**2+(bl[1]-((mousepos[1]+camPos[1])/ReScaleCoef-a["window"].Position[1]))**2)**0.5,0)
                                except AttributeError:
                                    pass
                                if dist <= 5:
                                    if a["window"].isThereTile(5)[0] == False:
                                        a["window"].placeTile(activetile,bl)
                                        a["window"].doRefresh = True
                                        a["window"].Root.doRefresh = True
                #============================= Section dediée à la manipulation de tuile ====================================
                if not cpos == None:
                    mousepos = cpos[:]
                    try:
                        activetile.Rotation = int(atan((mouseposcopy[1]-cpos[1])/-(mouseposcopy[0]-cpos[0]))*180/pi)
                    except ZeroDivisionError:
                        activetile.Rotation = 0
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                    cpos = mousepos[:]
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                    cpos = None
            #===================================== Section dediée aux layers ============================================
            elif type(activetile) == list and activetile[-2] == "layercreation":
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    activetile[0].Position[0], activetile[0].Position[1] = (mousepos[0]+camPos[0])//ReScaleCoef, (mousepos[1]+camPos[1])//ReScaleCoef
                    activetile[-1] = True
                #Actualise
                if activetile[-1] == True:
                    activetile[0].doRefresh = True
                    activetile[0].Root.doRefresh = True
                    activetile[0].Dimensions[0], activetile[0].Dimensions[1] = abs((mousepos[0]+camPos[0])//ReScaleCoef-activetile[0].Position[0]), abs((mousepos[1]+camPos[1])//ReScaleCoef-activetile[0].Position[1])
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    activetile[0].openAttrWindow()
                    activetile = None
                    LayerBox.SandBox.Buttons[-1].Name = "activelayer"
                    LayerBox.SandBox.Buttons[-1].Text = "onRender:str(self.Reference.Name)+' L'+str(self.Reference.Layer)"
                    MainEnv.seticon("none")
                    LayerBox.SandBox.doRefresh = True
                    LayerBox.doRefresh = True
            #============================================    Effects    =================================================
            elif type(activetile) == effect and type(a["window"]) == subLayer:
                if mouse_pressed[0] == True:
                    a["window"].placeEffect(activetile,((mousepos[0]+camPos[0])//ReScaleCoef-a["window"].Position[0],(mousepos[1]+camPos[1])//ReScaleCoef-a["window"].Position[1]),30,2)
                    a["window"].doRefresh = True
                    MainEnv.doRefresh = True
            #===========================================    Materiaux    ================================================
            elif type(activetile) == material:
                if isinstance(a["window"],subLayer):
                    if MainEnv.activeuicon == "none":       # Remet le curseur en brush
                        MainEnv.activeuicon = "material"+str(ceil(activetile.brushsize))
                        pygame.mouse.set_visible(True)
                    if mouse_pressed[0]:
                        if isinstance(a["window"],subLayer) == True:
                            a["window"].placeMaterial(mousecoordinates,ceil(activetile.brushsize))
                    elif event.type == pygame.MOUSEWHEEL:
                        activetile.brushsize += event.y/2
                        if activetile.brushsize > 3:
                            activetile.brushsize = 3
                        elif activetile.brushsize < 0:
                            activetile.brushsize = 0
                        MainEnv.activeuicon = "material"+str(ceil(activetile.brushsize))
                    elif event.type == pygame.KEYUP and (event.key == pygame.K_UP or event.key == pygame.K_DOWN): #Changement de variation
                        if event.key == pygame.K_UP:
                            activetile.var -= 1
                        else:
                            activetile.var += 1
                        activetile.var %= 4
                        MainEnv.matcursorcolorchange((activetile.color[0]*(activetile.var+1)//4+63*(3-activetile.var),activetile.color[1]*(activetile.var+1)//4+63*(3-activetile.var),activetile.color[2]*(activetile.var+1)//4+63*(3-activetile.var)))
                    elif mouse_pressed[2]:
                        if isinstance(a["window"],subLayer) == True:
                            a["window"].placeMaterial(mousecoordinates,ceil(activetile.brushsize),True) # Brush size
                elif isinstance(a["window"],toolBox):   # Remet le curseur en normal si mouse sur toolbox
                    MainEnv.activeuicon = "none"
                    pygame.mouse.set_visible(True)
            #=======================================     Scroll sandbox    ==============================================
            if event.type == pygame.MOUSEWHEEL:
                if type(a["window"]) == internalUI:
                    a["window"].LocalScroll += event.y*30
                    a["window"].Root.doRefresh = True
                    a["window"].doRefresh = True
                if mouse_pressed[0] == False and mouse_pressed[1] == False and mouse_pressed[2] == False and type(activetile) == tileRef:
                    Tilebox.SandBox.scrollTile(event.y)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if type(a["button"]) == toolButton:
                    a["button"].execScript()
            #===================================== Section dediée aux inputs ============================================
            elif event.type == pygame.KEYDOWN and type(inputref) == list and inputref[-2] == "input":
                #Del character
                if event.key == pygame.K_BACKSPACE and not inputref[0] == "":
                    inputref[0] = inputref[0][:-1]
                    try:
                        inputref[1].Root.doRefresh = True
                    except AttributeError:
                        inputref[1].doRefresh = True
                #Validation
                elif event.key == pygame.K_RETURN:
                    inputref[3].Theme = themes["ClassicAlt"]
                    inputref[3].doRefresh = True
                    inputref[3].Root.doRefresh = True
                    try:
                        inputref[1].Root.doRefresh = True
                    except AttributeError:
                        inputref[1].doRefresh = True
                    if not type(inputref[1]) == toolButton:
                        # ----------------------------------------- INPUT - Enter ----------------------------------------
                        try:
                            inputref[1][inputref[2]] = ast.literal_eval(inputref[0])
                        except SyntaxError as err:
                            raiseError(err)
                        except ValueError as err:
                            raiseError(err)
                    else:
                        inputref[1][inputref[2]] = inputref[0]
                    inputref = None
                else:
                    inputref[0] += event.unicode
                    try:
                        inputref[1].Root.doRefresh = True
                    except AttributeError:
                        inputref[1].doRefresh = True
            #=====================================    QuickMenuSelection    ============================================
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_KP0 or event.key == pygame.K_INSERT:
                    if not QuickMenu in toolboxes:
                        QuickMenu.Position = mousepos[:2]
                        toolboxes.append(QuickMenu)
                        QuickMenu.QuickMenuSelect(r'here\Assets\Editor\Placeables')
                        activetile = None
                    MainEnv.activeuicon = "none"
                    pygame.mouse.set_visible(True)
                elif event.key == pygame.K_KP1 or event.key == pygame.K_END:
                    if not MaterialsQuickMenu in toolboxes:
                        MaterialsQuickMenu.Position = mousepos[:2]
                        toolboxes.append(MaterialsQuickMenu)
                        MaterialsQuickMenu.QuickMenuSelect2(r'here\Assets\Editor\Placeables\Materials')
                        activetile = None
                    MainEnv.activeuicon = "none"
                    pygame.mouse.set_visible(True)
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_KP0 or event.key == pygame.K_INSERT:
                    if QuickMenu in toolboxes:
                        toolboxes.pop(toolboxes.index(QuickMenu))
                elif event.key == pygame.K_KP1 or event.key == pygame.K_END:
                    if MaterialsLQuickMenu in toolboxes:
                        toolboxes.pop(toolboxes.index(MaterialsLQuickMenu))
                    if MaterialsQuickMenu in toolboxes:
                        toolboxes.pop(toolboxes.index(MaterialsQuickMenu))
            #=====================================        END                ============================================
            elif event.type == pygame.QUIT:
                running = False
        if not inputref == None:
            inputref[3]["Text"] = inputref[0]
            inputref[3].Root.doRefresh = True
        elif inputref == None and type(inputref) == list and not inputref[0] == "":
            MainEnv.Dimensions = ast.literal_eval(inputref[0])
        renderGraphics()
        if uwuland == True:
            RenderPreview()

def renderGraphics():
    scrn.fill((40,40,40))
    #Render edt environment
    scrn.blit(MainEnv.Graphics(),(MainEnv.Position[0]*ReScaleCoef-camPos[0],MainEnv.Position[1]*ReScaleCoef-camPos[1]))
    #Render toolboxes
    for k in toolboxes:
        scrn.blit(k.Graphics(),k.Position)
    #Render tile preview
    if type(activetile) == tileRef:
        toblit = activetile.returnGraphics(True)
        toblit = pygame.transform.scale(toblit,(toblit.get_size()[0]*ReScaleCoef,toblit.get_size()[1]*ReScaleCoef))
        scrn.blit(toblit,(mousepos[0]-toblit.get_size()[0]//2,mousepos[1]-toblit.get_size()[1]//2))
    if not testt == None:
        scrn.blit(testt,(0,0))
    a = MainEnv.geticon()
    a[0].set_colorkey((0,0,0))
    scrn.blit(a[0],(mousepos[0]+a[1][0],mousepos[1]+a[1][1]))

def whatsOnMyMouse(pos):
    inputref = None
    for k in range(len(toolboxes)-1,-1,-1):
        l = toolboxes[k]
        if pos[0] >= l.Position[0] and pos[0] <= l.Position[0]+l.Dimensions[0]:
            if pos[1] >= l.Position[1] and pos[1] <= l.Position[1]+l.Dimensions[1]:
                return l.whatsOnMyMouse((pos[0]-l.Position[0],pos[1]-l.Position[1]))
    for l in MainEnv.subLayers.returnOrderedlistInv(camPos[2]):
        if pos[0]+camPos[0] >= l.Position[0]*ReScaleCoef and pos[0]+camPos[0] <= (l.Position[0]+l.Dimensions[0])*ReScaleCoef:
            if pos[1]+camPos[1] >= l.Position[1]*ReScaleCoef and pos[1]+camPos[1] <= (l.Position[1]+l.Dimensions[1])*ReScaleCoef:
                return {"window":l,"button":None}
    return {"window":None,"button":None}

def addBox(box,n):
    if n == 0:
        for l in [PropBox,LayerBox]:
            try:
                toolboxes.pop(toolboxes.index(l))
            except ValueError:
                pass
    if box in toolboxes:
        return
    else:
        toolboxes.append(box)

def raiseError(txt_):
    try:
        a = toolboxes.index(ErrorBox)
        toolboxes.pop(a)
    except ValueError:
        pass
    ErrorBox.Buttons = []
    ErrorBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":25,"Theme":themes["Error"],"Position":[365,8],"Text":"╳","Script":"self.closeSourceWindow()"})
    ErrorBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":18,"Theme":themes["Error"],"Position":[3,40],"Text":str(txt_),"Script":"self.closeSourceWindow()"})
    toolboxes.append(ErrorBox)

def refreshAll(obj):
    for l in obj.Buttons:
        l.doRefresh = True
        obj.doRefresh = True
        if type(obj) == internalUI:
            obj.Root.doRefresh = True

def RenderPreview():
    global testt, LoadedW
    start = time.perf_counter()
    RoomPreview = RoomModel({"Dimensions":MainEnv.Dimensions[:]})
    for k in MainEnv.subLayers.returnOrderedlist():
        RoomPreview.addRoomLayer(k)
    RoomPreview = LoadingRoom(RoomPreview)
    LoadedW = RoomPreview.continue_conversion()
    p1 = time.perf_counter()
    fov = packFOV(RequestScheme,pi/2.2,pi/4,0,RequestScheme.shape)
    p2 = time.perf_counter()
    assert LoadedW[5][0].dtype == numpy.int8, "problèmeu de détype >:c"
    assert LoadedW[6].dtype == numpy.uint8, "problèmeu de détype >:c"
    print("Size:",sys.getsizeof(LoadedW[5][0]))
    a = renderVisuals(LoadedW[0],LoadedW[1],LoadedW[2],LoadedW[3],LoadedW[4],LoadedW[5],LoadedW[6],LoadedW[7],LoadedW[8],fov,fov.shape,240,120,0,223,numpy.full((fov.shape[0],fov.shape[1],8),255,numpy.uint8),15,0.0,0.0)
    testt = pygame.surfarray.make_surface(a[:,:,:3])
    testt = pygame.transform.scale(testt,(windowLength,windowHeight))
    #with open(location+"\\arrayreport.txt","w") as k:
    #    k.write(str(fov))
    p3 = time.perf_counter()
    print("s1:",p1-start,"s2:",p2-p1,"s3:",p3-p2)

def act():
    global uwuland
    uwuland = True

#Défini les thèmes graphiques
themes["Classic"] = gTheme({"BorderColor":(0,255,0),"FontColor":(0,255,0),"BorderWidth":2,"Color":(50,30,0),"Font":"SimSun","ButtonBorderWidth":1,"ButtonFont":"SimSun"})
themes["QuickMenu"] = gTheme({"BorderColor":(255,255,0),"FontColor":(0,255,255),"BorderWidth":2,"Color":(0,15,40),"Font":"Segoe UI Symbol","ButtonBorderWidth":1,"ButtonFont":"Segoe UI Symbol"})
themes["QuickMenu2"] = gTheme({"BorderColor":(0,255,255),"FontColor":(255,255,0),"BorderWidth":2,"Color":(0,40,15),"Font":"Segoe UI Symbol","ButtonBorderWidth":1,"ButtonFont":"Segoe UI Symbol"})
themes["Error"] = gTheme({"BorderColor":(255,0,0),"FontColor":(255,0,0),"BorderWidth":3,"Color":(55,0,0),"Font":"SimSun","ButtonBorderWidth":1,"ButtonFont":"SimSun"})
themes["InputStandBy"] = gTheme({"BorderColor":(0,255,255),"FontColor":(0,255,255),"BorderWidth":3,"Color":(0,55,55),"Font":"SimSun","ButtonBorderWidth":0,"ButtonFont":"Segoe UI Symbol"})
themes["ClassicAlt"] = gTheme({"BorderColor":(0,255,0),"FontColor":(0,255,0),"BorderWidth":1,"Color":(50,30,0),"Font":"SimSun","ButtonBorderWidth":0,"ButtonFont":"Segoe UI Symbol"})
themes["SpritePreview"] = gTheme({"BorderColor":(0,255,0),"FontColor":(0,255,0),"BorderWidth":1,"Color":(50,30,0),"Font":"SimSun","ButtonBorderWidth":3,"ButtonFont":"Segoe UI Symbol"})
themes["SpritePreviewSelect"] = gTheme({"BorderColor":(250,100,0),"FontColor":(250,100,0),"BorderWidth":1,"Color":(50,30,0),"Font":"SimSun","ButtonBorderWidth":3,"ButtonFont":"Segoe UI Symbol"})
#Défini les interfaces
Tilebox = toolBox({"Position":[windowLength-250,0],"Dimensions":[250,windowHeight//1.5],"Theme":themes["Classic"],"Name":"Tilebox"})
PreviewBox = toolBox({"Position":[0,0],"Dimensions":[windowLength-250,windowHeight],"Theme":themes["Classic"],"Name":"PreviewBox"})
PropBox = toolBox({"Position":[windowLength-250,windowHeight//1.5],"Dimensions":[250,windowHeight//6],"Theme":themes["Classic"],"Name":"RoomProps"})
LayerBox = toolBox({"Position":[windowLength-250,windowHeight//1.5],"Dimensions":[250,windowHeight//6],"Theme":themes["Classic"],"Name":"LayerBox"})
MainBox = toolBox({"Position":[windowLength-250,windowHeight//1.5+windowHeight//6],"Dimensions":[250,windowHeight-windowHeight//1.5-windowHeight//6],"Theme":themes["Classic"],"Name":"MainBox"})
DefRotationPls = toolBox({"Position":[windowLength//2-100,windowHeight//2-50],"Dimensions":[200,100],"Theme":themes["Classic"],"Name":"SetRot"})
MaterialsLQuickMenu = toolBox({"Position":[0,0],"Dimensions":[35,250],"Theme":themes["QuickMenu2"],"Name":""})
MaterialsQuickMenu = toolBox({"Position":[0,0],"Dimensions":[130,250],"Theme":themes["QuickMenu2"],"Name":""})
QuickMenu = toolBox({"Position":[0,0],"Dimensions":[130,300],"Theme":themes["QuickMenu"],"Name":""})
LayerCreationBox = toolBox({"Position":[windowLength//2-200,windowHeight//2-100],"Dimensions":[400,200],"Theme":themes["Classic"],"Name":"Layer attributes"})
ErrorBox = toolBox({"Position":[windowLength//2-200,windowHeight//2-50],"Dimensions":[400,100],"Theme":themes["Error"],"Name":"Error"})
#Défini la sous interface de la tilebox
Tilebox.addSandBox({"Position":[Tilebox.Theme.BorderWidth-1,40],"Dimensions":[Tilebox.Dimensions[0]-Tilebox.Theme.BorderWidth*2+2,windowHeight//1.5],"Theme":themes["ClassicAlt"]})
toolboxes.append(Tilebox), toolboxes.append(MainBox), toolboxes.append(PropBox)
PropBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":25,"Theme":themes["Classic"],"Position":[215,8],"Text":"╳","Script":"self.closeSourceWindow()"})
PropBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":23,"Theme":themes["Classic"],"Position":[5,45],"Text":"RoomSize:","Script":""})
PropBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":23,"Theme":themes["ClassicAlt"],"Position":[120,41],"Text":"["+str(windowLength//ReScaleCoef)+","+str(windowHeight//ReScaleCoef)+"]","Script":"self.inputText(True,'Dimensions')","StoredReference":MainEnv})
PropBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":23,"Theme":themes["Classic"],"Position":[5,75],"Text":"RoomLayers:","Script":""})
PropBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":23,"Theme":themes["ClassicAlt"],"Position":[150,71],"Text":"3","Script":"self.inputText()"})
Tilebox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":25,"Theme":themes["Classic"],"Position":[215,8],"Text":"╳","Script":"self.closeSourceWindow()"})
Tilebox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":25,"Theme":themes["Classic"],"Position":[175,8],"Text":"↖","Script":r'self.openFolder("here\Assets\Editor\Placeables")'})
Tilebox.Buttons[-1].execScript()
MainBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":25,"Theme":themes["ClassicAlt"],"Position":[215,6],"Text":"","Script":""})
MainBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":25,"Theme":themes["ClassicAlt"],"Position":[185,6],"Text":"","Script":""})
MainBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":40,"Theme":themes["ClassicAlt"],"Position":[8,30],"Text":"","Script":""})
MainBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":25,"Theme":themes["ClassicAlt"],"Position":[40,42],"Text":"Props","Script":"addBox(PropBox,0)"})
MainBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":25,"Theme":themes["ClassicAlt"],"Position":[123,42],"Text":"sLayers","Script":"addBox(LayerBox,0)"})
MainBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":25,"Theme":themes["ClassicAlt"],"Position":[5,77],"Text":"Effects","Script":""})
MainBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":25,"Theme":themes["ClassicAlt"],"Position":[123,77],"Text":"Tiles","Script":"addBox(Tilebox,1)"})
MainBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":25,"Theme":themes["ClassicAlt"],"Position":[5,123],"Text":"Preview","Script":"RenderPreview()"})
LayerBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":25,"Theme":themes["ClassicAlt"],"Position":[215,6],"Text":"","Script":"self.closeSourceWindow()"})
LayerBox.addButton({"SubFolder":"","Name":"","GraphicType":"Unicode","FontSize":25,"Theme":themes["ClassicAlt"],"Position":[185,6],"Text":"","Script":'self.addLayer()'})
LayerBox.addSandBox({"Position":[LayerBox.Theme.BorderWidth-1,40],"Dimensions":[LayerBox.Dimensions[0]-LayerBox.Theme.BorderWidth*2+2,windowHeight//1.5],"Theme":themes["ClassicAlt"]})

launchEditor()