session_num=$1
iter=$2
set -x
nsml submit -v YOUR_ID/nipa_inpaint/$session_num $iter
