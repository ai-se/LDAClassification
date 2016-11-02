## 

- Different feature engineerings like just tf
- Different feature engineerings like tf+l2normrow
- Different feature engineerings like hash + tf + l2normrow
- Different feature engineerings like hash + tfidf + l2normrow


    pre="stem"
    sel="hash"
    fea="tf"
    norm="l2row"
    n_feature=4000
    label,dict=readfile_binary(filename=filepath+filename+filetype,thres=thres,pre=pre)
    dict=np.array(dict)
    dict=make_feature(dict,sel=sel,fea=fea,norm=norm,n_features=n_feature)
    
    pre = "stem"
    sel = "tfidf"
    fea = "tf"
    norm = "l2row"
    n_feature = 4000
    label, dict = readfile_binary(filename=filepath + filename + filetype, thres=thres, pre=pre)
    dict = np.array(dict)

    dict, voc = make_feature_voc(dict, sel=sel, fea=fea, norm=norm, n_features=n_feature)

    return dict, label, voc