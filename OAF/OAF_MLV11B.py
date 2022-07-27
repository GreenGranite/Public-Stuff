
from sqlalchemy import column
import OAF_MLFs_V11 as oaf
import datetime as dt
import pandas as pd 
import numpy as np
from pathlib import Path
from sklearn import neural_network as skl_nn
import sklearn.model_selection as skl_ms
import sklearn.metrics as skl_metrics
import matplotlib.pyplot as plt


# V11A (version 1.1A)
# Series 1 simple Sklearn MLP/ Nerual Networks

# 1) ML Estimate Returns for the month
# 2) takes top n Highest estimated returns
# 3) weight the porfolio according to the n Highest estimated returns
 
# A) Now incorporate Seeds for repeatability & Cross Validation Implementations 

idxs = pd.IndexSlice
dfname = "mldfV4C2"
names = ["AccRet","df_mcap_spx","df_P2BR_spx"]



mldf = pd.read_csv(r"Data In/{}.csv".format(dfname)).set_index(["Unnamed: 0","Unnamed: 1"])
tmbool = pd.read_csv(r"Data In/tmbool.csv").set_index(["Dates"])
AccRet = pd.read_csv(r"Data In/AccRet.csv").set_index(["Dates"])
MCap = pd.read_csv(r"Data In/df_mcap_spx.csv").set_index(["Dates"])
PtBR = pd.read_csv(r"Data In/df_P2BR_spx.csv").set_index(["Dates"])

MCap[tmbool == False]=np.NaN
PtBR[tmbool == False]=np.NaN
AccRet[tmbool == False]=np.NaN

# dfdict = oaf.MassReadCSV(names)
# dfdict = oaf.MassFormatting(tmbool,dfdict,True)

# AccRet = dfdict.pop(names[0])
# MCap = dfdict.pop(names[1])
# PtBR = dfdict.pop(names[2])


tval = int(np.round(tmbool.shape[0]*0.6))
tmbool_other = tmbool.iloc[tval:,:]
AccRet_other = AccRet.iloc[tval:,:]
MCap_other = MCap.iloc[tval:,:]
PtBR_other = PtBR.iloc[tval:,:]

tval = int(np.round(tmbool_other.shape[0]*0.5))
tmbool_test = tmbool_other.iloc[:tval,:]
tmbool_res = tmbool_other.iloc[tval:,:]
AccRet = AccRet_other.iloc[:tval,:]
AccRetRes = AccRet_other.iloc[tval:,:]
MCap = MCap_other.iloc[:tval,:]
MCapRes = MCap_other.iloc[tval:,:]
PtBR = PtBR_other.iloc[:tval,:]
PtBRRes = PtBR_other.iloc[tval:,:]

[mldf_train,mldf_other] = oaf.SplitData(tmbool,mldf,0.6)
[mldf_test,mldf_result] = oaf.SplitData(tmbool_other,mldf_other,0.5)

n = int(5)

idxs = pd.IndexSlice

# AccRet = AccRet.iloc[tval:,:]

# df_abl_test = mldf_test.loc[idxs[:,:],"Announcement Present"].unstack()
# df_abl_test[df_abl_test.isna()]=0

mldf_train[mldf_train["Next Month Return"]==0]=np.NaN
mldf_train.dropna(inplace=True)
#spliting up the data and the target
data = mldf_train.iloc[:,:-1]
target = mldf_train.iloc[:,-1]

#find a mask with points that is Not NaN 
pddnotna = mldf_test.iloc[:,:-1].notna().all(1)
resdnotna = mldf_result.iloc[:,:-1].notna().all(1)


predictdata = mldf_test.fillna(0).iloc[:,:-1]
resultdata = mldf_result.fillna(0).iloc[:,:-1]

ff3fdata = mldf_other.fillna(0).iloc[:,:-1]
ff3fdnotna = mldf_other.iloc[:,:-1].notna().all(1)

Search = False

if Search:
    model = skl_nn.MLPRegressor(random_state=2)
    param_grid = {
        'alpha': [0.00001,0.0001, 0.0005, 0.001, 0.01, 0.1],
        # 'hidden_layer_sizes' :[(100,50,30,30)]
        'hidden_layer_sizes':[(100,),(100,50),(100,50,30),(100,50,30,30)]
    }
    cvc = skl_ms.TimeSeriesSplit()
    search = skl_ms.GridSearchCV(model,param_grid,verbose=4,scoring='neg_mean_squared_error',cv=cvc).fit(data,target)
    ans = search.best_params_
    print(ans)
    mlans = search.predict(predictdata)
    mlansres = search.predict(resultdata)
else:
    model = skl_nn.MLPRegressor(alpha=0.00001, hidden_layer_sizes=(100,50,30,30), random_state=2)
    model.fit(data,target)
    mlans = model.predict(predictdata)
    mlansres = model.predict(resultdata)
    mlansff3f = model.predict(ff3fdata)


    



#actual return values
#AccRet = mldf_test.iloc[:,-1]

# ML to create estimated returns
# Selection of RS 2 becase it is not "broken" to begin with

# #ML to create estimated returns
# # # Selection of RS 2 becase it is not "broken" to begin with


# #Now adding Cross Validation








##GENERATE N.M.R. ESTIMATE
#Estimated returns and raw ML predictions
EstRet = (pd.Series(mlans,index = mldf_test.index)*pddnotna).unstack()
mlans = pd.Series(mlans,index = mldf_test.index)
EstRet = pd.DataFrame(EstRet.values,tmbool_test.index,EstRet.columns)

EstRetRes = (pd.Series(mlansres,index = mldf_result.index)*resdnotna).unstack()
EstRetRes = pd.DataFrame(EstRetRes.values,tmbool_res.index,EstRetRes.columns)
mlansres = pd.Series(mlansres,index = mldf_result.index)

EstRetff3f = (pd.Series(mlansff3f,index = mldf_other.index)*ff3fdnotna).unstack()
EstRetff3f = pd.DataFrame(EstRetff3f.values,tmbool_other.index,EstRetff3f.columns)
mlansff3f = pd.Series(mlansff3f,index = mldf_other.index)

# fp = Path('{}/{}.csv'.format("Data Out","TestEstRet"))  
# fp.parent.mkdir(parents=True, exist_ok=True)  
# EstRetRes.to_csv(fp)


WghMat = oaf.WghMatCreateTopN(EstRet,n)
WghMatRes = oaf.WghMatCreateTopN(EstRetRes,n)
WghMatff3f = oaf.WghMatCreateTopN(EstRetff3f,n)

# ##CONVERT ESTIMATE TO WEIGHT MATRIX
# #Stratergy converting Raw returns to normal returns
# WghMat = EstRet.sort_index(axis=1)

# #remove all negative predictions
# WghMat[WghMat<=0]=0
# WghMat[WghMat.isna()]=0

# # #take top n results
# # WghMat = WghMat.T.sort_values(by=WghMat.index.tolist(),ascending=False)
# # WghMat.iloc[(n):,:]=0
# # WghMat = WghMat.T
# # WghMat.sort_index(axis=0,inplace=True)
# # WghMat.sort_index(axis=1,inplace=True)

# #WghMat *= df_abl_test

# WghMat = oaf.NRowlargest(WghMat,n)
# WghMat.sort_index(axis=0,inplace=True)
# WghMat.sort_index(axis=1,inplace=True)

# WghMat = WghMat.divide(WghMat.sum(axis=1,skipna=True,numeric_only=True),axis=0)

##GENERATE SPX EVENLY WEIGHTED MATRIX
SPXWgh = tmbool_test*1
SPXWgh = SPXWgh.divide(SPXWgh.sum(axis=1),axis=0)

SPXWghRes = tmbool_res*1
SPXWghRes = SPXWghRes.divide(SPXWghRes.sum(axis=1),axis=0)

SPXWghff3f = tmbool_other*1
SPXWghff3f = SPXWghff3f.divide(SPXWghff3f.sum(axis=1),axis=0)

outname0 = dfname + "0bpsOutput"
outname5 = dfname + "5bpsOutput"
outnameres0 = dfname + "0bpsOutputRes"
outnameres5 = dfname + "5bpsOutputRes"


out = oaf.CreateOutput("Results",WghMat,AccRet,EstRet,SPXWgh,0,ReturnDF=True,OutPutName=outname0)
out5 = oaf.CreateOutput("Data Out",WghMat,AccRet,EstRet,SPXWgh,5,ReturnDF=True,OutPutName=outname5)

outres = oaf.CreateOutput("Results",WghMatRes,AccRetRes,EstRetRes,SPXWghRes,0,ReturnDF=True,OutPutName=outnameres0)
outres5 = oaf.CreateOutput("Data Out",WghMatRes,AccRetRes,EstRetRes,SPXWghRes,5,ReturnDF=True,OutPutName=outnameres5)

ret1 = out5.head(-1)["Actual Earnings"]
ret2 = out5.head(-1)["Evenly Weighted SPX"]

retres1 = outres5.head(-3)["Actual Earnings"]
retres2 = outres5.head(-3)["Evenly Weighted SPX"]

print("\n\nV1.1 Stratergy: Neural Networks + Top {} Results only ".format(n))
print("Strategy with DataFrame (Test): {}".format(dfname))

[MeanAnnualRet, Sharpe, MaxDrawDown, MDDPosition] = oaf.CalcMeasurements(ret1)

print("\nEvenly Weighted SPX (Test)")
[MeanAnnualRet, Sharpe, MaxDrawDown, MDDPosition] = oaf.CalcMeasurements(ret2)

print("\nStrategy with DataFrame (Result): {}".format(dfname))

[MeanAnnualRet, Sharpe, MaxDrawDown, MDDPosition] = oaf.CalcMeasurements(retres1)

print("\nEvenly Weighted SPX (Result)")
[MeanAnnualRet, Sharpe, MaxDrawDown, MDDPosition] = oaf.CalcMeasurements(retres2)



FF3FVal = oaf.CalcFF3F(WghMat,AccRet,MCap,PtBR)

FF3FRes = oaf.CalcFF3F(WghMatRes.iloc[:-2,:],AccRetRes.iloc[:-2,:],MCapRes,PtBRRes.iloc[:-2,:])

FF3FAcc = oaf.CalcFF3F(WghMatff3f.iloc[:-2,:],AccRet_other.iloc[:-2,:],MCap_other,PtBR_other.iloc[:-2,:])

SPXWghOverall = tmbool_other*1
SPXWghOverall = SPXWghOverall.divide(SPXWghOverall.sum(axis=1),axis=0)


FF3FSPX = oaf.CalcFF3F(SPXWghOverall.iloc[:-2,:],AccRet_other.iloc[:-2,:],MCap_other,PtBR_other.iloc[:-2,:])
print(FF3FSPX)

ans = WghMat.max(1,skipna=True,numeric_only=True)

plt.figure(8)
plt.plot(100*ans)
plt.xticks(np.arange(10,105,12,int),np.arange(2014,2022,1,int))
plt.title("Max Stock Weight in Strategy")
plt.xlabel("Months")
plt.ylabel("% Weighting")
plt.show()

oaf.Plotting(out)
# oaf.Plotting(out5)
# plt.close('all')

oaf.Plotting(outres,True,dfname,"Data Out")
# oaf.Plotting(outres5)

oaf.Plotting(out5)
# oaf.Plotting(out5)
# plt.close('all')

oaf.Plotting(outres5,True,dfname,"Data Out")
# oaf.Plotting(outres5)

ans = WghMatRes.max(1,skipna=True,numeric_only=True)

plt.figure(8)
plt.plot(100*ans)
plt.xticks(np.arange(10,105,12,int),np.arange(2014,2022,1,int))
plt.title("Max Stock Weight in Strategy")
plt.xlabel("Months")
plt.ylabel("% Weighting")
plt.show()
figname = dfname + "figure7"
figpath = Path("{}\{}.png".format("Data Out",figname))
plt.savefig(fname=figpath)

FF3F = pd.DataFrame(data=np.stack([FF3FVal,FF3FRes,FF3FAcc],0),index=["FF3FVal","FF3FRes","FF3FAcc"],columns=["alpha","risk beta","size beta","value beta","R2 Score"])

fp = Path('{}/{}.csv'.format("Data Out","{}FF3F".format(dfname)))  
fp.parent.mkdir(parents=True, exist_ok=True)  
FF3F.to_csv(fp)

print(FF3F)

print("Fin")