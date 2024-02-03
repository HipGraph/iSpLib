#!/usr/bin/sh

# ^[0-9.]*\t.* Ctrl+Shift+L

# for i in amazon reddit protein reddit2 product mag; do
for i in protein product mag; do
    python cogdl-gcn-sparse.py $i isplib >> cogdl-gcn.txt
    # python dgl-gin-sparse.py $i isplib >> dgl-gin.txt
    # python dgl-gin-sparse.py $i isplib nopad >> dgl-gin-nopad.txt
    # python dgl-graphSAGE-sparse.py $i isplib gcn nopad >> dgl-sage-sum-nopad.txt
    # python dgl-graphSAGE-sparse.py $i isplib mean nopad >> dgl-sage-mean-nopad.txt
    # python dgl-graphSAGE-sparse.py $i isplib gcn >> dgl-sage-sum.txt
    # python dgl-graphSAGE-sparse.py $i isplib mean >> dgl-sage-mean.txt
    # python dgl-gcn-sparse.py $i isplib >> dgl-gcn.txt
done


# for i in amazon reddit protein reddit2 product mag; do
#     for j in pt2 pt1 isplib; do
#         python gcn-sparse.py $i $j >> gcn-v2.txt
#         python graphSAGE-sparse.py $i $j sum nopad >> graphSAGE-sum-nopad-v2.txt
#         python graphSAGE-sparse.py $i $j sum >> graphSAGE-sum-v2.txt
#         python graphSAGE-sparse.py $i $j mean nopad >> graphSAGE-mean-nopad-v2.txt
#         # python gin-sparse.py $i $j >> gin-v2.txt
#         # echo "Outer: $i, Inner: $j"
#     done
# done
