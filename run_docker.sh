# Insufficient number of arguments
if [ $# -lt 1 ]; then
    echo "Usage: ./run_docker.sh [run|exec|build|stop|remove]"
    exit 1
fi

case $1 in
    run)
        # Run the docker container
        docker run -v ./:/app/ --rm --gpus device=3 -d -it --name fol-container2 fol #device=0,1,$CUDA_VISIBLE_DEVICES
        ;;
    exec)
        # Execute the models inside the docker container
        docker exec -it fol-container2 bash      
        ;;
    build)
        # Build the docker
        docker build ./ -t fol
        ;;
    stop)
        # Stop the docker container
        docker stop fol-container2
        ;;
    remove)
        # Remove the docker container
        docker stop fol-container2 &&
        docker remove fol-container2
        ;;
    *)
        # Invalid argument
        echo "Invalid argument"
        ;;
esac