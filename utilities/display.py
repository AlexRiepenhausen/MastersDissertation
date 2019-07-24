import torch
import time
import re
import ndjson
import jsonlines
from tkinter import *
from collections import defaultdict

class Display():

    def __init__(self, path, filename, numsamples, houses=False):
        self.houses     = False
        self.path       = path
        self.filename   = filename
        self.numsamples = numsamples
        self.rewind     = False
        self.master     = None
        self.data       = self.readFromFile()  
        

    def run(self):
    
        index = self.getStartingPoint()
        
       # for i in range(0,5000):
           # self.data[i]['property_type'] = 'flat' 
       # self.writeToFile()
       # exit(0)
        
        while index < self.numsamples:
            
            #if self.data[index]['char_count'] == 1908:
                #self.master = Tk()
                #self.displayFile(index) 
                #self.writeToFile()  
            
            self.master = Tk()
            self.displayFile(index)      
            
            if self.rewind:
                if index != 0:
                    index = index-1
                self.rewind = False
            else:
                index = index+1
                
        self.writeToFile()
        
    
    def displayAnnotationInfo(self):
    
        property_type     = defaultdict(int)
        tenement_steading = defaultdict(int)
        exclusive_strata  = defaultdict(int)
        exclusive_solum   = defaultdict(int)
        common_strata     = defaultdict(int)
        common_solum      = defaultdict(int)
        additional_info   = defaultdict(int)  
        char_count        = defaultdict(int)                        
            
        for i in range(0, self.numsamples): 
            
            a = self.data[i]['property_type']
            
            if not self.houses:                
                b = self.data[i]['tenement_steading']
                c = self.data[i]['exclusive_strata']
                
            d = self.data[i]['exclusive_solum']
            
            if not self.houses:
                e = self.data[i]['common_strata']
                
            f = self.data[i]['common_solum']
            g = self.data[i]['additional_info']
            
            property_type[a]     += 1
            
            if not self.houses:
                tenement_steading[b] += 1
                exclusive_strata[c]  += 1
                
            exclusive_solum[d]   += 1
            
            if not self.houses:
                common_strata[e] += 1
                
            common_solum[f]      += 1
            additional_info[g]   += 1  
                      
            char_count[i] = self.data[i]['char_count']
                      
                      
        # histogram     
        print("-----------------------------   Property Type   -----------------------------\n")
        self.printDictionary(property_type)
        
        if not self.houses:
            print("----------------------------- Tenement Steading -----------------------------\n")   
            self.printDictionary(tenement_steading) 
            print("----------------------------- Exclusive Strata  -----------------------------\n")
            self.printDictionary(exclusive_strata)  
            
        print("-----------------------------  Exclusive Solum  -----------------------------\n")            
        self.printDictionary(exclusive_solum)   
        
        if not self.houses:
            print("-----------------------------   Common Strata   -----------------------------\n")
            self.printDictionary(common_strata) 
            print("-----------------------------    Common Solum   -----------------------------\n")   
            self.printDictionary(common_solum)  
               
        print("-----------------------------  Additional Info  -----------------------------\n")               
        self.printDictionary(additional_info) 
        
        
    def printDictionary(self, dictionary): 
        for item in dictionary:
            print(" {:70} {:4d}".format(item, dictionary[item])) 
        print("")       
         
                              
    def getStartingPoint(self):
        return int(input("Which file number do you want to start from? "))                              
                                                              
        
    def readFromFile(self):
        with open(self.filename) as f:  
            return ndjson.load(f)               
                
                
    def writeToFile(self):
        with open(self.filename, mode='w') as writer:
            ndjson.dump(self.data, writer)
        print("Saved new tags to file")
            
            
    def previous(self):
        self.rewind = True
        self.master.destroy
            
        
    def getKeywords(self):
        path = './data/w2v/training/dictionary/display.txt'
        keywords = list()
        label = 0
        for line in open(path, encoding="utf8"):
            keywords.append(line.replace('\n',''))
            label += 1
        return keywords
        
        
    def assignHighlights(self, t1):
    
        colours = ["red","brown","pink","blue","yellow","mauve","green","grey","purple","orange","violet","black","white"]
    
        keywords = self.getKeywords()
        
        for keyword in keywords:

            pos_start = '0.0'
            
            while True:   
                pos_start = t1.search(keyword, pos_start, END)
                if pos_start:
                    pos_end = pos_start.split('.')[0] + '.' + str(int(pos_start.split('.')[1]) + len(keyword))
                    behind = int(pos_start.split('.')[1])
                    if behind > 0:
                        behind = behind - 1
                        behind_index = pos_start.split('.')[0] + '.' + str(behind)
                        if t1.get(behind_index) == ' ':      
                            if keyword in colours:  
                                t1.tag_add(keyword, pos_start, pos_end)
                            else:
                                t1.tag_add("style", pos_start, pos_end)
                
                    pos_start = pos_end
                    
                else:
                    break
        
            
    def displayFile(self, index):
    
        property_type     = 'TBA'
        tenement_steading = 'TBA'
        exclusive_solum   = 'TBA'
        common_solum      = 'TBA'
        exclusive_strata  = 'TBA'
        common_strata     = 'TBA'
        additional_info   = 'TBA'
        char_count        = len(self.data[index]['text']) 
        
        if 'char_count' not in self.data[index]:
            self.data[index]['char_count'] = char_count         
                
        if 'property_type' in self.data[index]:
            property_type = self.data[index]['property_type'] 
            
        if 'exclusive_solum' in self.data[index]:
            exclusive_solum = self.data[index]['exclusive_solum']       
            
        if 'common_solum' in self.data[index]:
            common_solum = self.data[index]['common_solum']   
            
        if 'additional_info' in self.data[index]:
            additional_info = self.data[index]['additional_info']     
            
        if not self.houses:
        
          if 'exclusive_strata' in self.data[index]:
              exclusive_strata = self.data[index]['exclusive_strata']   
            
          if 'common_strata' in self.data[index]:
              common_strata = self.data[index]['common_strata']     
              
          if 'tenement_steading' in self.data[index]:
              tenement_steading = self.data[index]['tenement_steading']          
            
        # format the text to be displayed
        
        _ =     'ID                : ' + self.data[index]['id'] + '\n'
        _ = _ + 'Number            : ' + str(index)      + '\n' 
        _ = _ + 'Character Count   : ' + str(char_count) + '\n' 
        _ = _ + 'Property Type     : ' + property_type   + '\n' 
        
        if not self.houses:
            _ = _ + 'Tenement Steading : ' + tenement_steading + '\n' 
            _ = _ + 'Exclusive Strata  : ' + exclusive_strata + '\n' 
            
        _ = _ + 'Exclusive Solum   : ' + exclusive_solum + '\n' 
        
        if not self.houses:
            _ = _ + 'Common Strata     : ' + common_strata    + '\n'   
            
        _ = _ + 'Common Solum      : ' + common_solum    + '\n'                               
        _ = _ + 'Additional Info   : ' + additional_info + '\n\n'                
        _ = _ + self.data[index]['text'] + '\n\n' 
            
        displayedtext = _
        
        property_type_input     = StringVar() 
        tenement_steading_input = StringVar()
        exclusive_strata_input  = StringVar() 
               
        exclusive_solum_input   = StringVar() 
        common_strata_input     = StringVar()        
        common_solum_input      = StringVar()
        additional_info_input   = StringVar()
        
        top = Frame(self.master)
        bottom = Frame(self.master)
        
        top.pack(side=TOP)
        bottom.pack(side=BOTTOM)       
        
        t1 = Text(self.master, width=140, height=30, font="Arial 14")
        t1.config(state='normal')
        t1.insert(INSERT, displayedtext)

        t1.tag_config("style", background="gainsboro", font=("Arial", "14", "bold"))
        
        t1.tag_config("red",    background="red",         foreground="white")
        t1.tag_config("black",  background="black",       foreground="white")
        t1.tag_config("black",  background="white smoke", foreground="black")
        t1.tag_config("brown",  background="brown",       foreground="white")
        t1.tag_config("pink",   background="pink",        foreground="white")
        t1.tag_config("blue",   background="blue",        foreground="white")
        t1.tag_config("yellow", background="yellow",      foreground="red")
        t1.tag_config("mauve",  background="cornsilk2",   foreground="black")
        t1.tag_config("green",  background="green",       foreground="white")
        t1.tag_config("orange", background="orange",      foreground="black")
        t1.tag_config("grey",   background="grey",        foreground="white")
        t1.tag_config("purple", background="purple1",     foreground="white")
        t1.tag_config("violet", background="purple3",     foreground="white")
        
        self.assignHighlights(t1)
            
        t1.config(state=DISABLED)
        
        l1 = Label(self.master, text="Prop Type", font="Times 8")
        l2 = None  
        l3 = None 
        
        l4 = Label(self.master, text="Exc Solum", font="Times 8")   
        l5 = None 
        l6 = Label(self.master, text="Com Solum", font="Times 8")  
        l7 = Label(self.master, text="Add Info",  font="Times 8")  
        
        if not self.houses:
            l2 = Label(self.master, text="Tnmnt Std", font="Times 8") 
            l3 = Label(self.master, text="Exc Strat", font="Times 8")
            l5 = Label(self.master, text="Com Strat", font="Times 8")            
                
                
        e1 = Entry(self.master, textvariable = property_type_input) 
        e2 = None
        e3 = None
        
        e4 = Entry(self.master, textvariable = exclusive_solum_input)
        
        e5 = None
        
        e6 = Entry(self.master, textvariable = common_solum_input)  
        e7 = Entry(self.master, textvariable = additional_info_input)   
        
        if not self.houses:
            e2 = Entry(self.master, textvariable = tenement_steading_input)
            e3 = Entry(self.master, textvariable = exclusive_strata_input) 
            e5 = Entry(self.master, textvariable = common_strata_input)         
        
        b1 = Button(self.master, text="Prev", width=5, justify=LEFT, command=self.previous)
        b2 = Button(self.master, text="Next", width=5, justify=LEFT, command=self.master.destroy)
        b3 = Button(self.master, text="Save", width=5, justify=LEFT, command=self.writeToFile)
        
        t1.pack(in_=top) 
        
        l1.pack(in_=bottom, side=LEFT)
        e1.pack(in_=bottom, side=LEFT)
        
        if not self.houses:
            l2.pack(in_=bottom, side=LEFT)
            e2.pack(in_=bottom, side=LEFT)
            l3.pack(in_=bottom, side=LEFT)
            e3.pack(in_=bottom, side=LEFT)            
            
        l4.pack(in_=bottom, side=LEFT)        
        e4.pack(in_=bottom, side=LEFT)   
        
        if not self.houses:
            l5.pack(in_=bottom, side=LEFT)
            e5.pack(in_=bottom, side=LEFT)
            
        l6.pack(in_=bottom, side=LEFT)
        e6.pack(in_=bottom, side=LEFT)      
        l7.pack(in_=bottom, side=LEFT)
        e7.pack(in_=bottom, side=LEFT)                             
                
        b1.pack(in_=bottom, side=LEFT)
        b2.pack(in_=bottom, side=LEFT)
        b3.pack(in_=bottom, side=LEFT)
        
        self.master.protocol("WM_DELETE_WINDOW", quit)
        
        mainloop()
        
        pti = property_type_input.get()
        esi = exclusive_solum_input.get()    
        tns = None    
        csi = common_solum_input.get()     
        adi = additional_info_input.get()   
        est = None
        cst = None   
        
        if not self.houses:
            tns = tenement_steading_input.get()
            est = exclusive_strata_input.get()
            cst = common_strata_input.get()      
        
        default = True      
        
        if pti != '':
            print("Index: {}, ID: {}, Old property type: {}, New property type: {}".format(index, self.data[index]['id'], property_type, pti))
            self.data[index]['property_type'] = pti
            default = False
            
        if esi != '':
            print("Index: {}, ID: {}, Old exclusive solum: {}, New exclusive solum: {}".format(index, self.data[index]['id'], exclusive_solum, esi)) 
            self.data[index]['exclusive_solum'] = esi   
            default = False 
  
        if csi != '':
            print("Index: {}, ID: {}, Old common solum: {}, New common solum: {}".format(index, self.data[index]['id'], common_solum, csi)) 
            self.data[index]['common_solum'] = csi 
            default = False       
            
        if not self.houses:
        
            if tns != '':
                print("Index: {}, ID: {}, Old tenement steading: {}, New tenement steading: {}".format(index, self.data[index]['id'], tenement_steading, tns)) 
                self.data[index]['tenement_steading'] = tns   
                default = False  
        
            if est != '':
                print("Index: {}, ID: {}, Old exclusive strata: {}, New exclusive strata: {}".format(index, self.data[index]['id'], exclusive_strata, est)) 
                self.data[index]['exclusive_strata'] = est   
                default = False  

            if cst != '':
                print("Index: {}, ID: {}, Old common strata: {}, New common strata: {}".format(index, self.data[index]['id'], common_strata, cst)) 
                self.data[index]['common_strata'] = cst   
                default = False                  
                       
            
        if adi != '':
            print("Index: {}, ID: {}, Old common additional info: {}, New common additional info: {}".format(index, self.data[index]['id'], additional_info, adi))
            self.data[index]['additional_info'] = adi
            default = False  
            
        # assign default if everythig is TBA and no manual assignement has happened yet
        
        if not self.houses:
            pass
        else:
            if default == True and \
               property_type == 'TBA' and \
               exclusive_solum == 'TBA' and \
               common_solum == 'TBA' and \
               additional_info == 'TBA':
               
               print("Index: {}, ID: {}, Default assigned".format(index, self.data[index]['id']))
               
               self.data[index]['property_type'] = 'house'
               self.data[index]['exclusive_solum'] = 'edged red' 
               self.data[index]['common_solum'] = 'none'  
               self.data[index]['additional_info'] = 'none'
                