build: Dockerfile
	docker build -t alex-klein-mlhw1 .

run: build
	docker run --rm -it -v `pwd`:/home/jovyan/work mlhw1

terminal:
	docker run --rm -it -v `pwd`:/home/jovyan/work mlhw1 /bin/bash

clean:
	docker rmi alex-klein-mlhw1