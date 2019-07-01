import torch
import time
import ndjson
import jsonlines
from tkinter import *

class Display():

    def __init__(self, path, filename, numsamples):
        self.path       = path
        self.filename   = filename
        self.numsamples = numsamples
        self.rewind     = False
        self.master     = None
        self.data       = self.readFromFile()
        
  
    def run(self):
    
        index = self.getStartingPoint()
        
        while index < self.numsamples:
        
            self.master = Tk()
            self.displayFile(index)      
            
            if self.rewind:
                if index != 0:
                    index = index-1
                self.rewind = False
            else:
                index = index+1
                
        self.writeToFile()
        
                              
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

            
    def displayFile(self, index):
    
        property_type   = 'TBA'
        exclusive_solum = 'TBA'
        common_solum    = 'TBA'
        char_count      = len(self.data[index]['text']) 
        
        if 'char_count' not in self.data[index]:
            self.data[index]['char_count'] = char_count         
                
        if 'property_type' in self.data[index]:
            property_type = self.data[index]['property_type'] 
            
        if 'exclusive_solum' in self.data[index]:
            exclusive_solum = self.data[index]['exclusive_solum']       
            
        if 'common_solum' in self.data[index]:
            common_solum = self.data[index]['common_solum']    
            
        # format the text to be displayed
        
        _ =     'ID             : ' + self.data[index]['id'] + '\n'
        _ = _ + 'Number         : ' + str(index)      + '\n' 
        _ = _ + 'Character Count: ' + str(char_count) + '\n' 
        _ = _ + 'Property Type  : ' + property_type   + '\n' 
        _ = _ + 'Exclusive Solum: ' + exclusive_solum + '\n' 
        _ = _ + 'Common Solum   : ' + common_solum    + '\n\n'           
        _ = _ + self.data[index]['text'] + '\n\n' 
            
        displayedtext = _
        
        property_type_input   = StringVar() 
        exclusive_solum_input = StringVar() 
        common_solum_input    = StringVar()
        
        top = Frame(self.master)
        bottom = Frame(self.master)
        
        top.pack(side=TOP)
        bottom.pack(side=BOTTOM)
        
        t1  = Label(self.master, text=displayedtext, pady=10, anchor=W, justify=LEFT, font="Times 14",wraplength=1300,background='white')
        
        l1 = Label(self.master, text="Property Type", font="Times 14")
        l2 = Label(self.master, text="Exclusive Solum", font="Times 14") 
        l3 = Label(self.master, text="Common Solum", font="Times 14")   
        
        e1  = Entry(self.master, textvariable = property_type_input) 
        e2  = Entry(self.master, textvariable = exclusive_solum_input)
        e3  = Entry(self.master, textvariable = common_solum_input)
        
        b1  = Button(self.master, text="Prev", width=10, justify=LEFT, command=self.previous)
        b2  = Button(self.master, text="Next", width=10, justify=LEFT, command=self.master.destroy)
        b3  = Button(self.master, text="Save", width=10, justify=LEFT, command=self.writeToFile)
        
        t1.pack(in_=top) 
        
        l1.pack(in_=bottom, side=LEFT)
        e1.pack(in_=bottom, side=LEFT)
        l2.pack(in_=bottom, side=LEFT)
        e2.pack(in_=bottom, side=LEFT)
        l3.pack(in_=bottom, side=LEFT)
        e3.pack(in_=bottom, side=LEFT)
        
        b1.pack(in_=bottom, side=LEFT)
        b2.pack(in_=bottom, side=LEFT)
        b3.pack(in_=bottom, side=LEFT)
        
        self.master.protocol("WM_DELETE_WINDOW", quit)
        
        mainloop()
        
        pti = property_type_input.get()
        esi = exclusive_solum_input.get()        
        csi = common_solum_input.get()        
        
        if pti != '':
            print("Index: {}, ID: {}, Old property type: {}, New property type assigned: {}".format(index, self.data[index]['id'], property_type, pti))
            self.data[index]['property_type'] = pti
            
        if esi != '':
            print("Index: {}, ID: {}, Old exclusive solum: {}, New exclusive solum assigned: {}".format(index, self.data[index]['id'], exclusive_solum, esi)) 
            self.data[index]['exclusive_solum'] = esi    
  
        if csi != '':
            print("Index: {}, ID: {}, Old common solum: {}, New common solum assigned: {}".format(index, self.data[index]['id'], common_solum, csi)) 
            self.data[index]['common_solum'] = csi     
