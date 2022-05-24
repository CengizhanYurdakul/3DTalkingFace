DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if [ ! -d "$DIR/v4l2loopback" ]
then
    git clone https://github.com/alievk/v4l2loopback.git
    echo "--- Installing v4l2loopback (sudo privelege required)"
    cd v4l2loopback
    make && sudo make install
    sudo depmod -a
    cd ..
else
    cd $DIR/v4l2loopback
    git pull
    cd ..
fi

sudo modprobe v4l2loopback exclusive_caps=1 video_nr=9 card_label="TalkingHead"

xhost + 
docker run -it --runtime=nvidia --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --network=host --device="/dev/video0:/dev/video0" --device=/dev/video9:/dev/video9 -v ${PWD}:/app talkingface