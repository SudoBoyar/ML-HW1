build: Dockerfile
	docker build -t mlhw1 .

run: build
	docker run --rm -it -p 8888:8888 -v `pwd`:/home/jovyan/work mlhw1

terminal:
	docker run --rm -it -v `pwd`:/home/jovyan/work mlhw1 /bin/bash
