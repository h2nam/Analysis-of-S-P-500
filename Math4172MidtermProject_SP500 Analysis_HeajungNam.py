# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 14:51:11 2018

@author: Heajung
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pprint
from pyomo.environ import ConcreteModel, Suffix, Var,Objective, Constraint, NonNegativeReals, maximize, value
from pyomo.opt import SolverFactory

#--------------------------------------
# 1. Import and Clean-up the Data File
#--------------------------------------
#import the DEA_SP500.xlsx file 
sp = pd.read_excel('DEA_SP500.xlsx')
#set the index as company name 
sp = sp.set_index('Company Name')
#Data Clean-Up
#It was observed that there are data entries with null values.
#Companies such as Target Corp or Amazon.com.inc have null values for certain quarters since they measured the variable only at the certain period. 
#find which companies are with missing data 
#determine whether each row has any null value
anynull1 = sp.isnull().any(axis=1)
#add the boolean values by each company name to determine whether the company has any null value
anynull2 = anynull1.groupby('Company Name').sum()
#find list of company names with at least one row with null value
anynull3 = anynull2.index[anynull2 !=0]
#remove any companies with null values
spNew = sp.drop(anynull3)

#-------------------
# 2. Set Data Frame
#-------------------
#update the data file by extracting columns for 8 variables
spNew2 = spNew[['Receivables - Total', 'Long-Term Debt - Total','Capital Expenditures','Cost of Goods Sold','Revenue - Total','Earnings Per Share (Basic) - Excluding Extraordinary Items']]
#update the data file by calculating the mean values of each variables for each company
spNew2 = spNew2.groupby('Company Name').mean()

#--------------------------
# 3. Visualization of Data 
#--------------------------
sp.boxplot()

sp.hist(column='Receivables - Total', 
        color='blue')

sp.plot(kind='scatter',
        x='Fiscal Quarter',
        y='Revenue - Total',
        figsize=(10,10),
        ylim=(0,200000))

sp.plot(kind='scatter',
        x='Fiscal Year',
        y='Revenue - Total',
        figsize=(10,10),
        ylim=(0,200000))

#Time series graph of the variables
spNew3 = sp.groupby(['Fiscal Year'])['Fiscal Year','Receivables - Total', 'Long-Term Debt - Total','Capital Expenditures','Cost of Goods Sold','Revenue - Total','Earnings Per Share (Basic) - Excluding Extraordinary Items'].mean()
spNew3.plot()

spNew4 = sp.groupby(['Fiscal Year'])['Fiscal Year','Receivables - Total', 'Long-Term Debt - Total','Capital Expenditures','Cost of Goods Sold','Revenue - Total','Earnings Per Share (Basic) - Excluding Extraordinary Items'].mean()
I = pd.DataFrame(np.array(spNew3.values),columns=['Year','RV','LD','CE','COGS','Revenue','Earning'])
I = I.astype(dtype={'Year':'int64'})
I = I.set_index('Year')
T,N = I.shape            
r = I.pct_change()       #percentage changes
r.describe()             

fsize=(14,7)
I.plot(title='value',figsize=fsize)
r.plot(title='%Change',figsize=fsize)
r.plot.bar(figsize=fsize)
r.plot.barh(figsize=fsize)
(I/I.iloc[0]).plot(title='value cumsum',figsize=fsize)
(r+1).cumprod().plot(title='value cumsum',figsize=fsize)
pd.plotting.scatter_matrix(r,diagonal='kde',figsize=fsize)

#----------------------------------
# 4. Set Function for Optimization
#----------------------------------
def DEAs(nUnits, nInputs, nOutputs, Inputs, Outputs,M=1):
	def One(current):
		U = np.arange(nUnits)
		I = np.arange(nInputs)
		O = np.arange(nOutputs)
		model = ConcreteModel()
		model.dual = Suffix(direction=Suffix.IMPORT)
		model.t = Var(O,within=NonNegativeReals) #declare variables with domain
		model.w = Var(I,within=NonNegativeReals)
		model.obj = Objective(expr= sum(model.t[o]*Outputs[current,o] for o in O), sense=maximize) #using expression
		model.con1 = Constraint(U, rule = lambda mo,u : sum(mo.t[o]*Outputs[u,o] for o in O) <= (M if u==current else 1) * sum(mo.w[i]*Inputs[u,i] for i in I) ) #using rules for a set of constraints
		model.con2 = Constraint(expr = sum(model.w[i] * Inputs[current,i] for i in I) <= 1) #using expr for a single constraint
		eps = 0.0001
		model.con3 = Constraint(I, rule = lambda mo, i: model.w[i] >= eps)
		model.con4 = Constraint(O, rule = lambda mo, o: model.t[o] >= eps)        
		return model
	
	opt = SolverFactory("ipopt") #pick up solver conda install -c conda-forge ipopt
	results = []

	from pyomo.opt import SolverStatus, TerminationCondition # for checking solver status
	for current in np.arange(nUnits):
		m = One(current)
		print('\n\n\nCurrent Model Index: %d\n' % current)
		m.pprint()
		r = opt.solve(m)
		print('Solver Status: ',  r.solver.status)
		print('Solver Terminate: ', r.solver.termination_condition)
		#assert (r.solver.status == SolverStatus.ok) and (r.solver.termination_condition == TerminationCondition.optimal) #necessary!!!
		r.write()
		_t = np.zeros(nOutputs)
		_w = np.zeros(nInputs)
		_d = np.zeros(nUnits)
		for o in np.arange(nOutputs):
			_t[o] = value(m.t[o]) #unfortunately, pyomo only let us access variables values one at a time
		for i in np.arange(nInputs):
			_w[i] = value(m.w[i]) #unfortunately, pyomo only let us access variables values one at a time
		for u in np.arange(nUnits):
			_d[u] = m.dual[m.con1[u]]
		results.append({'eff':value(m.obj), 'out':_t, 'in':_w, 'dual':_d})
	return results

#------------------
# 5. Optimization
#------------------
#set data for input and output
ins = spNew2[['Receivables - Total', 'Long-Term Debt - Total','Capital Expenditures','Cost of Goods Sold']]
outs = spNew2[['Revenue - Total','Earnings Per Share (Basic) - Excluding Extraordinary Items']]
#extracting the unique company names
names = ins.index.unique()

#set values for the variables before proceed with the DEAs function
nUnits = len(names)                    #total number of the companies
nInputs = 4                            #number of input variables
nOutputs = 2                           #number of output variables
Inputs = np.array(ins.values)          #transform the values of input as list of arrays
Outputs = np.array(outs.values)        #transform the values of output as list of arrays 

#results for classical DEA, set M=1
results = DEAs(nUnits, nInputs, nOutputs, Inputs, Outputs,M=1)

#results for super DEA, set M>1
#results_s = DEAs(nUnits, nInputs, nOutputs, Inputs, Outputs,M=3)
#results_s = DEAs(nUnits, nInputs, nOutputs, Inputs, Outputs,M=5)
#results_s = DEAs(nUnits, nInputs, nOutputs, Inputs, Outputs,M=10)

#print out the result of calculations
print('\n\n\nSolution:')
pprint.pprint(results)

#----------------
# 6. The Results 
#----------------
#set the data frame of the result to varaible efficiency 
efficiency = pd.DataFrame(data=results)
#set the index of the result as the company names
efficiency = efficiency.set_index(names)
#display the calculated efficiencies for each company
efficiency['eff']

#-------------------------------------
# 7. Export the Results to Excel File
#-------------------------------------
efficiency = efficiency['eff']
writer = pd.ExcelWriter('output.xlsx')
efficiency.to_excel(writer, 'Sheet1')
writer.save()

#---------------------------------
# 8. Visualization of the Results 
#---------------------------------
fsize=(14,7)
#scatter plot 
efficiency.plot(title='Efficiency',figsize=fsize)
#bar chart
efficiency.plot.bar(figsize=fsize)
#scatter plot
pd.plotting.scatter_matrix(r,diagonal='kde',figsize=fsize)
#box-plot
efficiency.boxplot()

#statistics of the results
def db(v):print(v,' =\n', repr(eval(v)), '\n')
mean = efficiency.mean(); db('mean')
gmean = pow((efficiency+1).cumprod().iloc[-1],1.0/(T-1))-1; db('gmean')
std = efficiency.std();db('std')
corr = efficiency.corr();db('corr')