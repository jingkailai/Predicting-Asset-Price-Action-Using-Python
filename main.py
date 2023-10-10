# GUI packages
from tkinter import *
from tkinter import messagebox

# Regular packages
import sys
import datetime
from dateutil.relativedelta import relativedelta

# Our functions
from final_group import * # lai minxian
from compare_sort import * # gerald
from Lstm_Model import * # chen wei
from ARIMA_GARCH import * # li xiaomeng


class Application(Frame):
    def __init__(self, master = None):  #定义构造函数   是构造器，用来构造组建对象
                                   #master=None表示初始化的时候值是空
                                   #创建的时候将定义的master=root主窗口对象传进去了
        super().__init__(master)  #Frame是父的构造器，不主动调用是不会被调用的
                                #通过super()调用Frame的构造方法，同时把master传进去了
                                #super()代表的是父类的定义，而不是父类对象
        self.master = master  #增加一个属性
        self.pack()  #self是一个组件,需要调用布局管理器进行排布和显示
    
        self.enter_page_1()  #实现调用   

    def Clear(self):
        # clears the window for the next Page
        for widget in self.winfo_children():
                widget.destroy()

    def enter_page_1(self):
        self.Clear()
        
        # Username entry
        Label(self, text="Username").grid(row=0, column=0)
        self.e1 = Entry(self); self.e1.grid(row=0, column=1)
        # Expected Return Ratio entry
        Label(self, text="Expected Return Ratio").grid(row=1, column=0)
        self.e2 = Entry(self, text="Expected Return Ratio"); self.e2.grid(row=1, column=1)
        # Duration entry
        Label(self, text="Duration").grid(row=2, column=0)
        self.v1 = IntVar()
        Radiobutton(self, text="7 days", value=7, variable=self.v1).grid(row=2, column=1, sticky=W)
        Radiobutton(self, text="14 days", value=14, variable=self.v1).grid(row=2, column=2, sticky=W)
        Radiobutton(self, text="1 month", value=30, variable=self.v1).grid(row=2, column=3, sticky=W)
        Radiobutton(self, text="3 month", value=90, variable=self.v1).grid(row=3, column=1, sticky=W)
        Radiobutton(self, text="6 month", value=182, variable=self.v1).grid(row=3, column=2, sticky=W)
        Radiobutton(self, text="1 year", value=365, variable=self.v1).grid(row=3, column=3, sticky=W)
        # Risk Tolerance entry
        Label(self, text="Risk Tolerance").grid(row=4, column=0)
        self.v2 = StringVar()
        Radiobutton(self, text="Very low", value="Very low", variable=self.v2).grid(row=4, column=1, sticky=W)
        Radiobutton(self, text="Low", value="Low", variable=self.v2).grid(row=4, column=2, sticky=W)
        Radiobutton(self, text="Medium", value="Medium", variable=self.v2).grid(row=4, column=3, sticky=W)
        Radiobutton(self, text="High", value="High", variable=self.v2).grid(row=5, column=1, sticky=W)
        Radiobutton(self, text="Very high", value="Very high", variable=self.v2).grid(row=5, column=2, sticky=W)


        Button(self, text="next", command=self.enter_page_2).grid(row=6, column=1)

        Label(self, text="This is an useful bot to help you select a stock!").grid(row=7, column=0, columnspan=3)

        return

    def enter_page_2(self):

        # get the input from page 1
        self.username = self.e1.get()
        if not self.username:
            messagebox.showinfo(title='Error', message='Please input a username!')
            Button(self, text="back", command=self.enter_page_1).pack()

        try:
            self.ret_ratio = float(self.e2.get())
        except:
            messagebox.showinfo(title='Error', message='Unvalid expected return ratio input!')
            Button(self, text="back", command=self.enter_page_1).pack()

        self.duration = self.v1.get()
        if not self.duration:
            messagebox.showinfo(title='Error', message='Please choose a duration for your investment!')
            Button(self, text="back", command=self.enter_page_1).pack()

        self.tol_rsk = self.v2.get().lower()
        if not self.tol_rsk:
            messagebox.showinfo(title='Error', message='Please choose your tolerance risk!')
            Button(self, text="back", command=self.enter_page_1).pack()

        # ??
        self.Clear()
        m = Label(self, text="Your best investment strategy will be generated after a few seconds, please wait...")
        m.pack()

        # Backend process
        self.start_date = datetime.date.today() + relativedelta(years=-10)
        self.today_date = datetime.date.today()
        self.stockdf = stock_filter(start_date=self.start_date, end_date=self.today_date, User_Risk=self.tol_rsk)
        self.stockdf.set_index(["Stock Ticker"], inplace=True)
        if self.duration <= 30: # less than 30 days, turn to lstm model
            self.model = 'lstm'
            self.resultlist = [lstm_model(ticker, self.start_date, self.today_date, self.duration)
                                for ticker in self.stockdf.index]
        else: # more than 30 days, turn to arima model
            self.model = 'arima'
            self.resultlist = [ARIMA_GARCH_prediction(ticker, self.duration)
                                for ticker in self.stockdf.index]
        
        self.resultlist = compare_and_sort(self.resultlist) # top 5

        # page 2
        self.Clear()
        self.l1 = Label(self,text="Recomended Choices:"); self.l1.grid(row=0, column=0)

        textlist = ["First", "Second", "Third", "Fourth", "Fifth"]
        self.v = IntVar()
        for i in range(len(self.resultlist)):
            Label(self, text="{} Stock".format(textlist[i])).grid(row=i+1, column=0, sticky=EW)
            Label(self, text="{:s}".format(self.resultlist[i][0])).grid(row=i+1, column=1, sticky=EW)
            Label(self, text="Return ratio: {:.3f}%".format(self.resultlist[i][2]*100)).grid(row=i+1, column=2, sticky=EW)
            Label(self, text="Test accuracy: {:.3f}%".format(self.resultlist[i][1]*100)).grid(row=i+1, column=3, sticky=EW)
            self.r1 = Radiobutton(self, text="{} Stock".format(textlist[i]), value=i, variable=self.v)
            self.r1.grid(row=i+1, column=5, sticky=EW)

        Button(self, text="next", command=self.enter_page_3).grid(row=6, column=2, sticky=EW)

        return

    def enter_page_3(self):
        # get input from page 2
        self.choice = self.v.get()
        ticker, ac, rt, sharprate = self.resultlist[self.choice]
        rsk = self.stockdf.loc[ticker, 'Risk']

        # page 3
        self.Clear()
        self.label01 = Label(self, text="Details:")
        self.label01.grid(row=0, column=0)


        Label(self, text="Stock Names").grid(row=1, column=0)
        Label(self, text=ticker).grid(row=1, column=1, columnspan=4)

        Label(self, text="Return:").grid(row=2, column=0)
        Label(self, text="{:.3f}%".format(rt*100)).grid(row=2, column=1, columnspan=4)

        Label(self, text="Risk:").grid(row=3, column=0)
        Label(self, text=rsk).grid(row=3, column=1, columnspan=4)

        Label(self, text="Sharp Ratio:").grid(row=4, column=0)
        Label(self, text="{:.3f}".format(sharprate)).grid(row=4, column=1, columnspan=4)

        Label(self, text="Model Accuracy:").grid(row=5, column=0)
        Label(self, text="{:.3f}%".format(ac*100)).grid(row=5, column=1, columnspan=4)

        #显示图像
        global photo  #把photo声明成全局变量，如果是局部变量，本方法执行完毕后图像对象销毁，窗口显示不出图像
        photo = PhotoImage(file="./figure/"+self.model+"_pred_"+ticker+".png")  #创建图像对象
        self.label02 = Label(self, image=photo)  #通过image属性指定图片
        self.label02.grid(row=5, column=1, rowspan=3, columnspan=4)


        self.btn01 = Button(self, text="Back", command=self.back)  #self就是一个组件容器，继承了Frame
                                 #相当于Button的master是self,当前的组件容器
        self.btn01.grid(row=8, column=1)

        self.btn02 = Button(self, text="Exit", command=self.exit)  
        self.btn01.grid(row=8, column=3)

        return

    def back(self):
        messagebox.showinfo("返回上一页")   #如何返回上一页
        self.enter_page_2()

        return

    def confirm(self):
        messagebox.showinfo("测试","选择性别:"+self.v.get())
        return 


    def exit(self):
        messagebox.showinfo("Thanks for using this APP!")
        sys.exit()



if __name__=='__main__':
    root = Tk()  #创建根窗口对象
    root.geometry("600x300")  #尺寸
    root.title("Portfolio Proposal")
    app = Application(master=root)  #创建一个对象，给它传一些参数(这里传了master参数)
    root = mainloop()