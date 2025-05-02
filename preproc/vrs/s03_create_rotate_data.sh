IMAGE_DIR=$1
CURRENT_DIR=$2

RGB_IMAGE_DIR=$IMAGE_DIR/'rotated_rgb'
LEFT_IMAGE_DIR=$IMAGE_DIR/'rotated_left'
RIGHT_IMAGE_DIR=$IMAGE_DIR/'rotated_right'

RGB_ROTATED_IMAGE_DIR=$IMAGE_DIR/'rgb'
LEFT_ROTATED_IMAGE_DIR=$IMAGE_DIR/'left'
RIGHT_ROTATED_IMAGE_DIR=$IMAGE_DIR/'right'

# ##---------------------------------------------------
echo "copying " $RGB_IMAGE_DIR
cp -r $RGB_IMAGE_DIR $RGB_ROTATED_IMAGE_DIR

echo "copying " $LEFT_IMAGE_DIR
cp -r $LEFT_IMAGE_DIR $LEFT_ROTATED_IMAGE_DIR

echo "copying " $RIGHT_IMAGE_DIR
cp -r $RIGHT_IMAGE_DIR $RIGHT_ROTATED_IMAGE_DIR

##---------------------------------------------------
cd $CURRENT_DIR
echo $CURRENT_DIR

echo "rotating " $RGB_ROTATED_IMAGE_DIR
./rotate_images.sh $RGB_ROTATED_IMAGE_DIR &

echo "rotating " $LEFT_ROTATED_IMAGE_DIR 
./rotate_images.sh $LEFT_ROTATED_IMAGE_DIR & 

echo "rotating " $RIGHT_ROTATED_IMAGE_DIR
./rotate_images.sh $RIGHT_ROTATED_IMAGE_DIR &

echo "Rotation minions are live! They will work hard in the background. Be patient and do not kill them!"