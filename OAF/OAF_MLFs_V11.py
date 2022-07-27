import datetime as dt
import pandas as pd 
import numpy as np
from pathlib import Path
import sklearn
import matplotlib.pyplot as plt
import sklearn.linear_model as skllm

# V1.1
# Change Log:
# - Simplfied MassFormatting() function
# - MonthShrink() function added

def MassReadCSV(filenames : list, Dformat : str = '%Y-%m', DFolderIn = "Data In") -> dict:
    ##DESCRIPTION

    # Takes:
    #   - filenames     - a list of text(sting) of the csv files that needs to be open
    #                     example: we want to open:
    #                       - example1.csv
    #                       - example2.csv ...
    #                     input list filenames:
    #                       - filenames = ["example1","example2",...]
    #                     each file should ALL have:
    #                       - Indexes in the form of dates (corrisponding to Dformat; Default: "%Y-%m") with index column Label "Dates"
    #                       - Column Titles in the form of <Tickers>
    #                     example:(Dformat = %Y-%m" (Default))
    #                       - Dates   0111145D    APPL
    #                         2000-01 value       value
    #                         2001-11 value       value 
    #   - Dformat       - a string (comply to datetime strptime() function see datetime documentation for strptime) 
    #                     denoting the way time object is written
    #                     Default: '%Y-%m' -> 2000-01
    # Returns:
    #   - vardict       - a dictionary of files, each key corrisponds to the names given in list: filenames
    #                     and values corrisponding to the datafield in the key
    #               

    ## CODE
    vardict = {}
    for name in filenames:
        tempdf = pd.read_csv(r"{}/{}.csv".format(DFolderIn,name))

        tempdf.set_index("Dates",inplace=True)

        nidxlist = []
        for mydate in tempdf.index:
            mydtobj = dt.datetime.strptime(mydate,Dformat)
            nidxlist.append(mydtobj)
        tempdf = pd.DataFrame(data = tempdf.values, index = nidxlist, columns = tempdf.columns)      
        
        vardict[name]=tempdf
    return vardict

def MassFormatting(tmbool:pd.DataFrame ,dfdict: dict, applymask: bool = False, applynormalisation: bool = False, frac:float = 0.6) -> dict:
    ##DESCRIPTION

    #Takes:
    #   - tmbool      - the time-membership boolean dataframe "key"; 
    #                   this holds the information of time period &
    #                   ticker of concern, and also wether if the ticker 
    #                   is a member of SPX in the period                 
    #   - dfdict      - a dictionary containing all relavent data frame, 
    #                   each dataframe corrisponds to a specific field/ 
    #                   varaible of interest for ML 
    #   - applymask   - overloaded boolean input to determine whether to remove
    #                   values not in SPX at the time 
    #Returns:
    #   - dfdict      - the formatted dictionary, with all missing points filled
    #                   with Numpy NaN Objects and all extra point NOT in the tmbool 
    #                   key removed

    ##CODE

    tval = int(np.round(tmbool.index.shape[0]*frac))

    for dfs in dfdict.keys():
        cdf = pd.DataFrame(dfdict[dfs])#cdf = current dataframe
        cdf = cdf.reindex_like(tmbool)
        
        #sorting rows and columns
        cdf.sort_index(axis=0,inplace=True) 
        cdf.sort_index(axis=1,inplace=True)
 
        if applymask == True:  
            cdf[tmbool==False]=np.NaN

        if applynormalisation == True:
            tdf = cdf.iloc[:tval,:]
            maxval = tdf[tdf.isin([0,np.NaN,np.inf,-np.inf,True,False])==False].abs().max().max()
            if np.isnan(maxval)==False:
                cdf = cdf.divide(maxval)
        
        dfdict[dfs] = cdf
    return dfdict

def GenMLDF(tmbool: pd.DataFrame,Formatted_dfdict: dict):
    ##DESCRIPTION

    #Takes:
    #   - tmbool    - the time-membership boolean dataframe; this holds the information of time period & ticker of concern, and also wether if the ticker exist in the period
    #   - fdfdict   - a FORMATTED dictionary containing all relavent data frame, 
    #                 each dataframe corrisponds to a specific field of interest,
    #                 MUST be formatted by MassFormatting FUNCTION BEFORE application here   
    #Returns:
    #   - mldf      - the "data" dataframe for machine learning

    ##CODE
    mldf = pd.DataFrame(columns=Formatted_dfdict.keys())
    tempdict ={}

    for dfs in Formatted_dfdict.keys():
        cdf = pd.DataFrame(Formatted_dfdict[dfs]).stack(dropna=False)      #cdf = current dataframe
        tempdict[dfs] = cdf
 
    for dfs in Formatted_dfdict.keys():
        cdf =tempdict[dfs]
        mldf[dfs]=cdf

    return mldf

def CalcReturns(weightmat:pd.DataFrame,returnmat:pd.DataFrame,k:float=0) ->pd.Series:
    ##DESCRIPTION
    # returning a Series of Retruns, given a weighting and returns matrix

    #Takes:
    #   - weightmat     - the weighting matrix, index = time series data, column = tickers, values = the fractional weight of the portfolio at said time
    #   - returnmat     - the return matrix, same index AND columns a the weightmat; corrisponds to the actual (%) Return of the ticker at said time period
    #Returns: 
    #   - ActualRet     - the commission ajusted (%) Returns
    #   - TransCost     - the total transactional cost, for said time period

    ##CODE
    PortfolioRet = (weightmat*returnmat).sum(1,skipna=True)
    TransCost = (k/10000)*(weightmat-weightmat.shift(-1,axis=0,fill_value=0)).abs().sum(1,skipna=True)
    ActualRet = PortfolioRet-TransCost
    return ActualRet, TransCost

def SplitData(tmbool: pd.DataFrame,mldf: pd.DataFrame, frac: float,RetDat:bool = False):
    idxs = pd.IndexSlice
    n_train =int(tmbool.shape[0]*frac)

    indexobj = tmbool.index[n_train]
    mldf_train = mldf.loc[idxs[:indexobj,:],:]
    mldf_test = mldf.loc[idxs[indexobj:,:],:]
    if RetDat:
        return mldf_train, mldf_test,indexobj
    else:
        return mldf_train, mldf_test

# def GenTMBoolKey(startdate: dt.date, enddate: dt.date, Tickers: set,)

def SoftMax(x):
    ans = np.exp(x)/sum(np.exp(x)) 
    return ans 

# def PlotComparison(ErnRet: pd.Series, SPXRet: pd.Series):
#     tempdf = pd.concat([ErnRet,SPXRet])
    
#     out = pd.Series(index=ErnRet)
#     initval = 1
    
#     for k in np.arange(0,tempdf.shape[0],1,int):
#         initval = tempdf.iloc[k,:]+tempdf.iloc[k,:]*abs(initval)
#         out
#     for k in np.ara
#     ErnRet.abs()

def MonthShrink(df:pd.DataFrame,method:int=0)->pd.DataFrame:
    ##DESCRIPTION
    #   takes a DataFrame with a datetime index and returns a DataFrame which is compressed into datetime mothly "buckets"

    # Takes:
    #   - df        - Input DataFrame, the dataframe must have a DatetimeIndex Object as it's index
    #   - method    - selection of methods:
    #                   - 0 :   .any() method use for Bool values, indicates there is a TRUE value in the period
    #                   - 1 :   nanmean() method, use for numerical values, apply mean in a collumn-wise fashion skipping nan values
    # Gives:
    #   - out       - the compressed DataFrame into monthly buckets.


    if method == 0:
        out = df.to_period("M").groupby(level=0).any()
    if method == 1:
        out = df.to_period("M").groupby(level=0).apply(lambda x: np.nanmean(x,axis=0))
        out = pd.DataFrame(out.values.tolist(), index=out.index,columns=df.columns)
    return out

def CreateOutput(OutFolder: str, WghMat: pd.DataFrame, AccRet: pd.DataFrame,EstRet: pd.DataFrame,SPXWgh: pd.DataFrame, bps: float = 0, OutPutName: str = "Output", ReturnDF: bool = False, WghMatOut: bool = True):

    [Earnings,Transactions] = CalcReturns(WghMat,AccRet,bps)
    [EstEarnings,temp] = CalcReturns(WghMat,EstRet,bps)
    [SPXControl,temp] = CalcReturns(SPXWgh,AccRet,0)

    StratPortGrowth = (Earnings+1).cumprod()
    SPXPortGrowth = (SPXControl+1).cumprod()
    GrowthPDiff = (StratPortGrowth-SPXPortGrowth)/SPXPortGrowth

    StratAnnRet = (StratPortGrowth.shift(-12)-StratPortGrowth)/StratPortGrowth
    SPXAnnRet = (SPXPortGrowth.shift(-12)-SPXPortGrowth)/SPXPortGrowth
    PrefDiff = StratAnnRet-SPXAnnRet
    

    out = pd.concat([EstEarnings,Earnings,SPXControl,Transactions,StratPortGrowth,SPXPortGrowth,GrowthPDiff,StratAnnRet,SPXAnnRet,PrefDiff],axis=1)
    out = pd.DataFrame(out.values,out.index,columns=["Estimated Earnings","Actual Earnings","Evenly Weighted SPX","Transaction Costs","Strategy Portfolio Growth", "SPX Mean Portfolio Growth","Percentage Growth Difference","Strategy Annual Return","SPX Mean Annual Return","Annual Return Diference"])
    
    fp = Path('{}/{}.csv'.format(OutFolder,OutPutName))  
    fp.parent.mkdir(parents=True, exist_ok=True)  
    out.to_csv(fp)
    
    if WghMatOut:
        fp = Path('{}/WghMat.csv'.format(OutFolder))  
        fp.parent.mkdir(parents=True, exist_ok=True)  
        WghMat.to_csv(fp)

        fp = Path('{}/EstRet.csv'.format(OutFolder))  
        fp.parent.mkdir(parents=True, exist_ok=True)  
        EstRet.to_csv(fp)

    if ReturnDF == True:
        return out

def CalcMeasurements(rets, periods_per_year = 12):

    #Adaptation of existing code

    #rets = 1D array or pandas series
    n = len(rets)/periods_per_year   #no. of years
    cumrets = (1+rets).cumprod()   #cumulative returns

    #Mean Annual Return
    MeanAnnualRet = (cumrets[-1]**(1/n) - 1)

    #scale to average annual return and volatility
    Sharpe = (MeanAnnualRet)/(np.std(rets) * np.sqrt(periods_per_year))
   
    max_cumrets = cumrets.cummax()  #max previous cumret
    dd = 1 - cumrets/max_cumrets   #all drawdowns
    MaxDrawDown = np.max(dd)
    MDDPosition = np.argmax(dd)
    print("Results:\n - Mean Annual Returns: \t{} \n - Sharpe Ratio: \t\t{} \n - Maximum Drawdown: \t\t{}".format(MeanAnnualRet,Sharpe,MaxDrawDown))

    return MeanAnnualRet, Sharpe, MaxDrawDown, MDDPosition

def PastNValidValue(dfdict:dict,df_bool:pd.DataFrame,n:int)-> dict:
    if n>0:
        for df in dfdict.keys():
            tdf =pd.DataFrame(dfdict[df])

            #set value to be the one previous 
            tdf[df_bool==False]=np.NaN 
            tdf = tdf.fillna(method="ffill").shift(1)
            
            dfdict[df] = tdf
        n-=1
        PastNValidValue(dfdict,df_bool,n)
    return dfdict

def NShiftRet(df:pd.DataFrame, SPeriod: int)->pd.DataFrame:
    out = (df-df.shift(SPeriod)).divide(df.shift(SPeriod))
    return out 

def Plotting(df:pd.DataFrame,savefigbool:bool=False,figname:str="NA",figlocname:str="NA"):

    #1st Figure: Portfolio Growth Over Time
    Skey = "Strategy Portfolio Growth"
    Mkey = "SPX Mean Portfolio Growth"
    plt.figure(0)
    # h1, = plt.plot(df[Skey],"b",linewidth=2)
    h1, = plt.plot(df[Skey],linewidth=2)
    h2, = plt.plot(df[Mkey],"r",linewidth=2)
    plt.title("Portfolio Growth Over Time")
    plt.xlabel("Months")
    plt.ylabel("Growth")
    plt.xticks(np.arange(10,105,12,int),np.arange(2014,2022,1,int))
    plt.axvline(x=df.index[0],color="k",linestyle="--")
    
    #2nd Figure: Percentage Growth Difference
    Skey ="Percentage Growth Difference"
    plt.figure(1)
    plt.plot(df[Skey]*100)
    plt.axline((0,0),slope=0,color="k",linestyle="--")
    #plt.fill_between(np.arange(0,df.shape[0],1,int),df[Skey],0,interpolate=True,color="c")
    plt.title(Skey)
    plt.xlabel("Months")
    plt.ylabel("% Difference")
    plt.xticks(np.arange(10,105,12,int),np.arange(2014,2022,1,int))
    plt.axvline(x=df.index[0],color="k",linestyle="--")

    #3rd Figure: Annual Returns
    Skey = "Strategy Annual Return"
    Mkey = "SPX Mean Annual Return"
    plt.figure(2)
    # h1, = plt.plot(df[Skey],"b",linewidth=2)
    h1, = plt.plot(df[Skey]*100,linewidth=2)
    h2, = plt.plot(df[Mkey]*100,"r",linewidth=2)
    plt.title("Annual Returns")
    plt.xlabel("Months")
    plt.ylabel("% Return")
    plt.xticks(np.arange(10,105,12,int),np.arange(2014,2022,1,int))
    plt.axvline(x=df.index[0],color="k",linestyle="--")

    #4th Figure: Annual Return
    Skey = "SPX Mean Annual Return"
    Mkey = "Strategy Annual Return"

    plt.figure(3)
    plt.scatter(x=df[Skey]*100,y=df[Mkey]*100)
    plt.axvline(color="k",linestyle="--")
    plt.axline((0,0),slope=0,color="k",linestyle="--")
    plt.axline((0,0),slope=1,color="b",linestyle="-")

    plt.title("Stratergy Returns Vs Even SPX Returns")
    plt.ylabel("Strategy % Returns")
    plt.xlabel("Even SPX % Returns")

    #5th Figure: Annula Return Difference
    Skey ="Annual Return Diference"
    plt.figure(4)
    plt.plot(df[Skey]*100)
    plt.axline((0,0),slope=0,color="k",linestyle="--")
    #plt.fill_between(np.arange(0,df.shape[0],1,int),df[Skey],0,interpolate=True,color="c")
    plt.title("% Points Difference on Returns")
    plt.xlabel("Months")
    plt.ylabel("% Points Difference")
    plt.xticks(np.arange(10,105,12,int),np.arange(2014,2022,1,int))
    plt.axvline(x=df.index[0],color="k",linestyle="--")

    #6th Figure: Portfolio Growth Over Time
    Skey = "Strategy Portfolio Growth"
    Mkey = "SPX Mean Portfolio Growth"
    plt.figure(5)
    # h1, = plt.plot(df[Skey],"b",linewidth=2)
    h1, = plt.plot(np.log(df[Skey]),linewidth=2)
    h2, = plt.plot(np.log(df[Mkey]),"r",linewidth=2)
    plt.title("Log Portfolio Growth Over Time")
    plt.xlabel("Months")
    plt.ylabel("Log Growth")
    plt.xticks(np.arange(10,105,12,int),np.arange(2014,2022,1,int))
    plt.axvline(x=df.index[0],color="k",linestyle="--")

    #7th Figure: Scatter between Estimated and Actual Earnings
    Mkey = "Actual Earnings"
    Skey = "Estimated Earnings"
    df1=df[df[Skey]<=10]
    df1=df1[df1[Skey]>=0]
    
    plt.figure(6)
    plt.scatter(x=df1[Skey]*100,y=df1[Mkey]*100)
    plt.axvline(color="k",linestyle="--")
    plt.axline((0,0),slope=0,color="k",linestyle="--")
    plt.axline((0,0),slope=1,color="b",linestyle="-")

    plt.title("Actual Vs Estimated Returns")
    plt.xlabel("Estimated % Returns")
    plt.ylabel("Actual % Returns")

    plt.show()
    
    #save pictures
    if savefigbool == True:
        for k in np.arange(0,7,1,int):
            plt.figure(k)
            accfigname = "{}figure{}".format(figname,k)
            figpath = Path("{}/{}.png".format(figlocname,accfigname))
            plt.savefig(fname=figpath)

def DomainNorm(df: pd.DataFrame,val: float = 10)-> pd.DataFrame:
    df = df/val
    df[df>1]=1
    return df

def NRowlargest(df:pd.DataFrame, n: int):
    
    
    for k in np.arange(0,df.shape[0],1,int):
        df.sort_values(by=[df.index[k]],axis=1,inplace=True,ascending=False)
        df.iloc[k,n:]=0
    
    return df

def WghMatCreateTopN(EstRet:pd.DataFrame,n:int):
    WghMat = EstRet.sort_index(axis=1)

    #remove all negative predictions
    WghMat[WghMat<=0]=0
    WghMat[WghMat.isna()]=0

    #take top n results
    WghMat = NRowlargest(WghMat,n)
    WghMat.sort_index(axis=0,inplace=True)
    WghMat.sort_index(axis=1,inplace=True)

    #normalise
    WghMat = WghMat.divide(WghMat.sum(axis=1,skipna=True,numeric_only=True),axis=0)
    return WghMat

def CalcFF3F(WghMat:pd.DataFrame,AccRet:pd.DataFrame,MCAP:pd.DataFrame,PTBR:pd.DataFrame):
    WT = WghMat.sort_index(0).sort_index(1)
    MCAP = MCAP.sort_index(0).sort_index(1).fillna(0)
    PTBR = PTBR.sort_index(0).sort_index(1).fillna(0)
    AccRet = AccRet.sort_index(0).sort_index(1).fillna(0)
 
    DIndex = WT.index
    DColumn = WT.columns

    # WT = WT.to_numpy()
    # WT = pd.DataFrame(WT,MCap.index,MCap.columns)

    # to create a psuedo bool dataframe as a "Mask"
    WT[WT>0]=1
    WT[WT<=0]=0
    
    
    # chaging the indexes and columns to make sure things work, assuming that indexes are identical!
    IdMCAP = pd.DataFrame(MCAP.values,DIndex,DColumn).replace(0,np.NaN)
    IdBTMR = pd.DataFrame(np.divide(np.ones(PTBR.shape),PTBR.values),DIndex,DColumn).replace(0,np.NaN)
    MCAP = pd.DataFrame(MCAP.values*WT.values,DIndex,DColumn).replace(0,np.NaN)
    BTMR = pd.DataFrame((np.divide(np.ones(PTBR.shape),PTBR.values))*WT.values,DIndex,DColumn).replace(0,np.NaN)
    AccRet = pd.DataFrame(AccRet.values,DIndex,DColumn).replace(0,np.NaN)
    StratRet = (AccRet*WghMat).sum(1,skipna=True,numeric_only=True)

    #finding the Medians relative to the portfolio to split into "big" vs "small"
    MCAPMedianVec = MCAP.median(axis=1,skipna=True,numeric_only=True).values
    MCAPMedian=[MCAPMedianVec for k in np.arange(0,WghMat.shape[1],1,int)]
    MCAPMedian= np.stack(MCAPMedian,axis=1)
    MCAPMedian=pd.DataFrame(MCAPMedian,DIndex,DColumn)
     
    BTMRMedianVec = BTMR.median(axis=1,skipna=True,numeric_only=True).values
    BTMRMedian = [BTMRMedianVec for k in np.arange(0,WghMat.shape[1],1,int)]
    BTMRMedian= np.stack(BTMRMedian,axis=1)
    BTMRMedian = pd.DataFrame(BTMRMedian,DIndex,DColumn)

    MCAPGT = ((MCAP>=MCAPMedian)*1*WT*AccRet).replace(0,np.NaN)
    MCAPLT = ((MCAP<MCAPMedian)*1*WT*AccRet).replace(0,np.NaN)
    BTMRGT = ((BTMR>=BTMRMedian)*1*WT*AccRet).replace(0,np.NaN)
    BTMRLT = ((BTMR<BTMRMedian)*1*WT*AccRet).replace(0,np.NaN)

    #finding the Medians relative to the index/ Market to split into "big" vs "small"
    IdMCAPMedianVec = IdMCAP.median(axis=1,skipna=True,numeric_only=True).values
    IdMCAPMedian=[IdMCAPMedianVec for k in np.arange(0,WghMat.shape[1],1,int)]
    IdMCAPMedian= np.stack(IdMCAPMedian,axis=1)
    IdMCAPMedian=pd.DataFrame(IdMCAPMedian,DIndex,DColumn)

    IdBTMRMedianVec = IdBTMR.median(axis=1,skipna=True,numeric_only=True).values
    IdBTMRMedian = [IdBTMRMedianVec for k in np.arange(0,WghMat.shape[1],1,int)]
    IdBTMRMedian= np.stack(IdBTMRMedian,axis=1)
    IdBTMRMedian = pd.DataFrame(IdBTMRMedian,DIndex,DColumn)
   
    IdMCAPGT = ((IdMCAP>=IdMCAPMedian)*1*AccRet).replace(0,np.NaN)
    IdMCAPLT = ((IdMCAP<IdMCAPMedian)*1*AccRet).replace(0,np.NaN)
    IdBTMRGT = ((IdBTMR>=IdBTMRMedian)*1*AccRet).replace(0,np.NaN)
    IdBTMRLT = ((IdBTMR<IdBTMRMedian)*1*AccRet).replace(0,np.NaN)

    # find weighted avaerage of the "big" and "small" groups according to the portfolio weighting
    MCAPWBMean = (MCAPGT*WghMat).sum(1).divide(((MCAP>=MCAPMedian)*1*WghMat).sum(1))
    MCAPWSMean = (MCAPLT*WghMat).sum(1).divide(((MCAP<MCAPMedian)*1*WghMat).sum(1))
    BTMRWBMean = (BTMRGT*WghMat).sum(1).divide(((BTMR>=BTMRMedian)*1*WghMat).sum(1))
    BTMRWSMean = (BTMRLT*WghMat).sum(1).divide(((BTMR<BTMRMedian)*1*WghMat).sum(1))
    
    # find the mean of the "big" and small groups in the Index
    IdMCAPWBMean = IdMCAPGT.mean(1)
    IdMCAPWSMean = IdMCAPLT.mean(1)
    IdBTMRWBMean = IdBTMRGT.mean(1)
    IdBTMRWSMean = IdBTMRLT.mean(1)

    # weighted "Small Minus Big"
    MCAPWBMS = MCAPWSMean - MCAPWBMean
    # weighted "High Minus Low"
    BTMRWBMS = BTMRWBMean - BTMRWSMean
    IdMCAPWBMS = IdMCAPWBMean - IdMCAPWSMean
    IdBTMRWBMS = IdBTMRWBMean - IdBTMRWSMean
    
    # Weighted "Big Minus Small" Relative to the Index
    # Tom's feedback, Not needed potential
    RMCAPWBMS = MCAPWBMS #(MCAPWBMS - IdMCAPWBMS).divide(IdMCAPWBMS)
    RBTMRWBMS = BTMRWBMS #(BTMRWBMS - IdBTMRWBMS).divide(IdBTMRWBMS)

    # Market Risk Primium note that Risk Free Rate Assumed to be 0
    IdMRP=AccRet.mean(1,numeric_only=True,skipna=True).fillna(0)
    Ones=pd.Series(np.ones(WghMat.shape[0]),DIndex)

    Data = pd.concat([Ones,IdMRP,RMCAPWBMS,RBTMRWBMS],axis=1).fillna(0,inplace=False)

    fp = Path('{}/{}.csv'.format("Data Out","JustTest"))  
    fp.parent.mkdir(parents=True, exist_ok=True)  
    Data.to_csv(fp)

    StratRet.fillna(0,inplace=True)

    Xt = Data.values.T
    X = Data.values
    y = StratRet.values
    
    w = np.linalg.pinv(Xt.dot(X)).dot(Xt.dot(y))
    pdiff=X.dot(w)-y
    score=pdiff.T.dot(pdiff)

    FF3F = np.append(w,score)
    

    # Lreg = skllm.LinearRegression()
    # Lreg.fit(Data,StratRet)
    # FF3F = Lreg.coef_
    # score = Lreg.score(Data,StratRet)
    # FF3F=np.append(FF3F,score)

    # fp = Path('{}/{}.csv'.format("Data Out","JustTest"))  
    # fp.parent.mkdir(parents=True, exist_ok=True)  
    # StratRet.to_csv(fp)
    
    # fp = Path('{}/{}.csv'.format("Data Out","JustTest2"))  
    # fp.parent.mkdir(parents=True, exist_ok=True)  
    # Data.to_csv(fp)

    return FF3F
    

    
    

  

    # fp = Path('{}/{}.csv'.format("Data Out","MTMBool"))  
    # fp.parent.mkdir(parents=True, exist_ok=True)  
    # MTMBool.to_csv(fp)

    # fp = Path('{}/{}.csv'.format("Data Out","PTMBool"))  
    # fp.parent.mkdir(parents=True, exist_ok=True)  
    # PTMBool.to_csv(fp)

    
    
    

    


        