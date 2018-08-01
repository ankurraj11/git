import pandas as pd
import numpy as np
from bson.objectid import ObjectId
from datetime import datetime
from datetime import date, datetime, time, timedelta
import pymongo
from pymongo import MongoClient


## function to get the metrics for comparison of Grids
def computations(pre, current):
    
    cprpPre = sum(pre['preOutlay']) / (sum(pre['preGrp']))
    cprpCurrent = sum(current['outlay']) / (sum(current['grp']))
        
    return(cprpPre, cprpCurrent)


##connection with the server
client = MongoClient(
    'mongodb://sagacito:sagacito123@star-cluster-shard-00-00-davlt.mongodb.net:27017,' \
    'star-cluster-shard-00-01-davlt.mongodb.net:27017,' \
    'star-cluster-shard-00-02-davlt.mongodb.net:27017/starBuildDev?ssl=true&replicaSet=Star-Cluster-shard-0&authSource=admin')

db = client['starBuildDev']
proposalGrid = db.proposalGridTxn
proposalBrief = db.proposalBriefTxn

##fetch _ids 
#proposalGridFrame = pd.DataFrame(list(proposalGrid.find({}, {'proposalId': 1, 'advertiserGroup.name': 1, 'timeBand': 1, '_id': 1})))

##fetch data for current grid
currentGrid = proposalGrid.find_one({"_id":ObjectId("5a3a445c86b5a96492e68cf2")})  ## Proposal Id will filter 

advertiserGroup = currentGrid['advertiserGroup']['name']
currentImpactInRegular = currentGrid['impactInRegular']
proposalId = currentGrid['proposalId']

currentBrief = proposalBrief.find_one({'proposalId':proposalId})
createdAt = currentBrief['createdAt']
print(type(currentImpactInRegular))
list1 = []

for i in range(0, len(currentGrid['productFrame'])):

    TVR_ch_TG = currentGrid['productFrame'][i]['tvrChannelPrTG']
    final_price = currentGrid['productFrame'][i]['priceFinal']
    FCT = currentGrid['productFrame'][i]['fctFinal']  
    outlay = (final_price * FCT) / 10
    grp = (FCT * TVR_ch_TG) / 10
    isCurrentImpact = currentGrid['productFrame'][i]['isImpactChannel']


    list1.append([advertiserGroup, TVR_ch_TG, final_price, FCT, grp, outlay, createdAt, currentImpactInRegular,isCurrentImpact])


current = pd.DataFrame(list1, columns=['advertiserGroup', 'TVR_ch_TG', 'final_price','FCT', 'grp' ,'outlay', 'createdAt', 'currentImpactInRegular','isCurrentImpact'])
current['impact'] = (current['isCurrentImpact'].apply(lambda x: 1 if x == 'Impact' else 0))
current['regular'] = (current['isCurrentImpact'].apply(lambda x: 1 if x == 'Regular' else 0))

##fetch data from the precedence grid as per the current grid advertiser

precedence_adv = pd.DataFrame(list(proposalGrid.find({'advertiserGroup.name': advertiserGroup}, {'_id': 1})))

precedence = pd.DataFrame()

for i in range(0, len(precedence_adv)):
    preGrid = proposalGrid.find_one({'_id': ObjectId(precedence_adv['_id'][i])})

    preProposalId = preGrid['proposalId']
    preAdvertiser = preGrid['advertiserGroup']['name']
    preImpactInRegular = preGrid['impactInRegular']

    preBrief = proposalBrief.find_one({'proposalId':preProposalId})

    preCreatedAt = preBrief['createdAt']
    list2 = []

    if (preGrid['status'] == 'approved'):
        for j in range(0, len(preGrid['productFrame'])):

            preTVR_ch_TG = preGrid['productFrame'][j]['tvrChannelPrTG']
            preFinal_price = preGrid['productFrame'][j]['priceFinal']
            preFCT = preGrid['productFrame'][j]['fctFinal']
            isPreImpact = preGrid['productFrame'][j]['isImpactChannel']

            preGrp = (preFCT * preTVR_ch_TG) / 10
            preOutlay = (preFinal_price * preFCT) / 10

            list2.append([preAdvertiser, preTVR_ch_TG, preFinal_price, preFCT, preGrp , preOutlay, preCreatedAt, preImpactInRegular, isPreImpact])

    new_df = pd.DataFrame(list2, columns=['preAdvertiser', 'preTVR_ch_TG', 'preFinal_price', 'preFCT', 'preGrp', 'preOutlay', 'preCreatedAt', 'preImpactInRegular','isPreImpact'])
    precedence = pd.concat([precedence,new_df])

precedence['impact'] = (precedence['isPreImpact'].apply(lambda x: 1 if x == 'Impact' else 0))
precedence['regular'] = (precedence['isPreImpact'].apply(lambda x: 1 if x == 'Regular' else 0))

    
## Finding the precedence to compare with
precedence_groupby = precedence.groupby('preCreatedAt')
precedence_sorted = sorted(precedence_groupby, reverse=True)
keys = precedence_groupby.groups.keys()

for i in range(1,len(keys)):
    frame = precedence_sorted[i][1]
    if(frame['regular'].sum() == 0 ):
        continue
    else:
        precedence = precedence_sorted[i][1]
        break

print(current , precedence)

## Another way to find the right precedence:
'''
precedence_adv = pd.DataFrame(list(proposalGrid.find({'advertiserGroup.name': advertiserGroup}, {'_id': 1},{'createdAt':1})))
precedence_temp = precedence_adv.groupby('createdAt')
precedence_sorted = sorted(precedence_temp, reverse=True)

keys = precedence_sorted.groups.keys()


for i in range(0, len(keys)-1):
    frame2 = precedence_sorted[i][1]
    frame = proposalGrid.find_one({'_id': ObjectId(frame2['_id'])}) ## or Using Created at
    if(frame['status']=='approved'): 
        if(frame['regular'].sum() == 0 ):
            continue
        else:
            precedence = precedence_sorted[i][1]
            break
''' 
## guardRail Comparision

if (list(set(current['currentImpactInRegular'])) == True):
    if (current['regular'].sum() == 0):
        print("No CPRP Gaurdrail required")
    elif (current['impact'].sum() == 0):
        precedence2 = precedence[precedence['regular']==1]
        CPRP_ch_prec, CPRP_ch_current = computations(precedence2, current)
        if CPRP_ch_current <= 0.99 * (CPRP_ch_prec):
            print(1,"Gaurdrail Not Met")
        else:
            print(1, "Gaurdrail Met")
    else:
        if (list(set(precedence['preImpactInRegular'])) == True):
            CPRP_ch_prec, CPRP_ch_current = computations(precedence, current)
            if CPRP_ch_current <= 0.99 * (CPRP_ch_prec):
                print(2,"Gaurdrail Not Met")
            else:
                print(2, "Gaurdrail Met")
            
        else:
            precedence2 = precedence[precedence['regular']==1]
            current2 = current[current['regular'] == 1]
            CPRP_ch_prec, CPRP_ch_current = computations(precedence2, current2)
            if CPRP_ch_current <= 0.99 * (CPRP_ch_prec):
                print(3,"Gaurdrail Not Met")
            else:
                print(3, "Gaurdrail Met")
            
    
else:
    if (current['regular'].sum() == 0):

        print("No CPRP Gaurdrail required")
        
    else:
        precedence2 = precedence[precedence['regular']==1]
        current2 = current[current['regular'] == 1]
        CPRP_ch_prec, CPRP_ch_current = computations(precedence2, current2)
        if CPRP_ch_current <= 0.99 * (CPRP_ch_prec):
            print(4,"Gaurdrail Not Met")
        else:
            print(4, "Gaurdrail Met")
        
  


