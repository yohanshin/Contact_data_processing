DIR=$1

for szFile in $DIR/*.jpg
do 
    convert "$szFile" -rotate 90 $DIR/"$(basename "$szFile")" ; 
done