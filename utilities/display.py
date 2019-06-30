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
    
        colour_txt = 'TBA'
        config_txt = 'TBA'
            
        if 'colour' in self.data[index]:
            colour_txt = self.data[index]['colour'] 
            
        if 'configuration' in self.data[index]:
            config_txt = self.data[index]['configuration']       
        
        displayedtext = "ID: " + self.data[index]['id'] + '\n' + \
                        "Number: " + str(index) +"\n" + \
                        "Colour: " + colour_txt + '\n' + \
                        "Config: " + config_txt + '\n\n' + \
                        self.data[index]['text'] + "\n"
        
        col_input = StringVar() 
        con_input = StringVar() 
        
        top = Frame(self.master)
        bottom = Frame(self.master)
        
        top.pack(side=TOP)
        bottom.pack(side=BOTTOM)
        
        lb  = Label(self.master, text=displayedtext, pady=10, anchor=W, justify=LEFT, font="Times 14",wraplength=1300,background='white')
        
        cll = Label(self.master, text="Colour", font="Times 14")
        cnk = Label(self.master, text="Config", font="Times 14")   
        
        cl  = Entry(self.master,text="Colour ", textvariable=col_input) 
        cn  = Entry(self.master,text="Config ", textvariable=con_input)
        
        bt1  = Button(self.master, text="Prev",  width=10, justify=LEFT, command=self.previous)
        bt2  = Button(self.master, text="Next",  width=10, justify=LEFT, command=self.master.destroy)
        bt3  = Button(self.master, text="Save",  width=10, justify=LEFT, command=self.writeToFile)
        
        lb.pack(in_=top) 
        cll.pack(in_=bottom, side=LEFT)
        cl.pack(in_=bottom, side=LEFT)
        cnk.pack(in_=bottom, side=LEFT)  
        cn.pack(in_=bottom, side=LEFT)  
        bt1.pack(in_=bottom, side=LEFT)
        bt2.pack(in_=bottom, side=LEFT)
        bt3.pack(in_=bottom, side=LEFT)
        
        self.master.protocol("WM_DELETE_WINDOW", quit)
        
        mainloop()
        
        col = col_input.get()
        if col != '':
            print("Index: {}, ID: {}, Old colour: {}, New colour assigned: {}".format(index, self.data[index]['id'], colour_txt, col_input.get()))
            self.data[index]['colour'] = col
            
        con = con_input.get()
        if con != '':
            print("Index: {}, ID: {}, Old config: {}, New config assigned: {}".format(index, self.data[index]['id'], config_txt, con_input.get())) 
            self.data[index]['configuration'] = con    
  
 
