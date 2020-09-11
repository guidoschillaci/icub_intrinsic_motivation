# in Macos do not forget to run in a separate terminal
# socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\"
# to display icub sim and GUIs
# xhost + 127.0.0.1
xhost +local:root

if [ -z "$1" ]
  then
    echo "No argument supplied. I need an ip address of the client pc, or just localhost."
    exit 1
fi

ipclient=$1

export DOCKER_CONTAINER_NAME=icub_container
if [ ! "$(docker ps -q -f name=${DOCKER_CONTAINER_NAME})" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=${DOCKER_CONTAINER_NAME})" ]; then
        echo "Cleaning up existing container named ${DOCKER_CONTAINER_NAME}"
        # cleanup
        docker rm $DOCKER_CONTAINER_NAME
    fi
    echo "Connecting to new container named ${DOCKER_CONTAINER_NAME}"
    # run your container 
    docker run -it --rm \
      -e DISPLAY=host.docker.internal:0 \
      --name "$DOCKER_CONTAINER_NAME" \
      --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
      --volume="/Users/guido/Documents/code:/code/:rw"  \
      -w /code \
      guidoski/icub:tf2-nogpu bash -c 'export YARP_DATA_DIRS=$YARP_DATA_DIRS:/code/icub_intrinsic_motivation/yarp/apps/ &&  bash'
else
    echo "Connecting to existing container named ${DOCKER_CONTAINER_NAME}"
    # run your container 
    docker exec -it \
      -e DISPLAY=host.docker.internal:0 \
      "$DOCKER_CONTAINER_NAME" \
      bash  -c 'export YARP_DATA_DIRS=$YARP_DATA_DIRS:/code/icub_intrinsic_motivation/yarp/apps/ && bash'
fi

#-e DISPLAY=docker.for.mac.host.internal:0 \


