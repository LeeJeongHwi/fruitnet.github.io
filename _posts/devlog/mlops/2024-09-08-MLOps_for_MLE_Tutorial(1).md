---
layout: post
title: MLOps) MLOps for MLE Tutorial (1) - Database
description: >
  MLOps for MLE Tutorial Database Part
sitemap: false
categories: [devlog, mlops]
hide_last_modified: true
related_posts:
  - ./2024-08-25-MLOps_춘추전국시대.md
  - ./2024-09-01-WSL2_WindowDocker.md
---

# MLOps) MLOps for MLE Tutorial (1) - Database

본격적으로 셋팅한 WSL2를 활용해서 MLOps Tutorial을 하고자한다.



## Docker PostgreSQL Run

해당 튜토리얼에서는 `PostgreSQL` 을 사용해서 Database를 구성

```shell 
$ docker run -d --name postgres-server -p 5432:5432 -e POSTGRES_USER=leepic -e POSTGRES_PASSWORD=password -e POSTGRES_DATABASE=mydatabase postgres:14.0
```

* `-d` : detach 형태로 실행, 컨테이너를 종료할 때 `ctrl+c`로 실행하게 되면 빠져나오게 됨
* `--name` : 컨테이너 이름 설정
* `-p` : 포트설정, Host:container 로, 현재 `5432:5432`로 설정했는데, host부에 있는 포트로 외부에서 container의 포트에 연결할 수 있다.
* `-e` : Container를 구성하는 필요한 환경변수 설정, Image 마다 필요한 환경변수가 다른데, docker-hub에서 컨테이너 설정하는 과정에서 필수로 요구하는 환경변수들이 있다. postgres의 경우 `POSTGRES_PASSWORD`르 필수로 설정해주어야한다
* `postgres:14.0` : image를 지정하는 문구, image:tag 로 이루어져있음, 일반적으로 tag는 버전을 의미함

### Table Creation

필요한 package 설치 

* `pip install pandas, psycopg2-binary, scikit-learn`

그 후 db Connection Code 만들기

~~~python
//file: table_creation.py
import psycopg2 # PostgreSQL Connector Import

if __name__:"__main__":
    db_connector = psycopg2.connect(
        user=DB_config.DB_USER,
        password=DB_config.DB_PASSWORD,
        host="localhost",
        database=DB_config.DB_DATABASE,
        port=DB_config.DB_PORT
    )
~~~

> DB_config는 Appendix 참조

위에서 container를 만들면서 설정해준 USER, PASSWORD, HOST, DATABASE, PORT를 설정해서 Connector를 만듬

그리고 IRIS 데이터 type에 맞게 Create Table code 작성

~~~python
sepal length (cm)    float64
sepal width (cm)     float64
petal length (cm)    float64
petal width (cm)     float64
target                 int64
dtype: object

def create_table(db_connector):
    create_table_query=f"""
    CREATE TABLE IF NOT EXISTS iris_data(
        id SERIAL PRIMARY KEY,
        timestamp timestamp,
        sepal_length float8,
        sepal_width float8,
        petal_length float8,
        petal_width float8,
        target int
    );
    """
    
    with db_connector.cursor() as cur:
        cur.execute(create_table_query)
        db_connector.commit()
    print("Create Table for IRIS DATA") # 없어도됨
~~~

사실 블로그에 `cursor`나 `commit, execute` 와 같은 코드들도 설명해야하나 고민했지만.. 이번에는 MLOps를 위한 것이니까, 필요한 것만 설명하고 기록하기로 했다.

매번 shell로 접속하기 귀찮아서 Shell script로 만들어서 실행시켜서 확인한다

![image-20240908154807856](./../../../images/2024-09-08-MLOps_for_MLE_Tutorial(1)/image-20240908154807856.png)

해당 shell script에도 정보가 담겨있기에 gitignore에 추가해주자..

그리고 data를 생성해서 넣는 과정이다.

```python
#//file: "data_generator.py"
def get_data():
    X, y = load_iris(return_X_y=True, as_frame=True)
    df = pd.concat([X, y], axis="columns")
    rename_rule = {
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
    }
    df = df.rename(columns=rename_rule)
    return df

def insert_data(db_connector, data):
    insert_row_query = f"""
    INSERT INTO iris_data (timestamp, sepal_length, sepal_width, petal_length, petal_width, target) VALUES 
    (NOW(), {data.sepal_length}, {data.sepal_width}, {data.petal_length}, {data.petal_width}, {data.target});
    """
    with db_connector.cursor() as cur:
        cur.execute(insert_row_query)
        db_connector.commit()

def generate_data(db_connector, df):
    i = 0
	len_df = len(df)
    while i<len_df:
        print()
        insert_data(db_connector, df.sample(1).squeeze())
        i+=1
        time.sleep(1) # DB 과부하 방지
```

매 초마다 df에서 sample 1개를 데이터를 가져와서 INSERT하는 코드이다.

원래는 `while True:`로 계속 받는 것으로 무한루프로 구성되어 있지만, 현재는 데이터 길이만큼 넣고 종료하는 것으로 구성했다.

해당 data-generator를 dockerfile로 만들어서 image화 시킨다

```dockerfile
FROM amd64/python:3.9-slim

RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/app

RUN pip install -U pip &&\
    pip install scikit-learn pandas psycopg2-binary

COPY data_generator.py data_generator.py
COPY config.py config.py

ENTRYPOINT ["python", "data_generator.py", "--db-host"]

CMD ["localhost"]
```

* `FROM`: 가져올 image
* `RUN`: 실행시킬 명령어
* `WORKDIR`: Working Directory (바로 실행되는 지점)
* `COPY` : 파일 복사 `COPY [local] [container]`
* `ENTRYPOINT` : 프로그램의 진입점, 컨테이너가 수행하게 될 실행명령을 정의
  * 컨테이너가 무슨일을 하는지 최종 단계를 정의
* `CMD` : 추가 인자를 받아서 실행시킴

여기서 ENTRYPOINT와 CMD의 차이를 둘 때에는 다음과 같은 관점으로 보자

* 컨테이너가 수행될 때 변경되지 않을 실행 명령은 `ENTRYPOINT`로 정의
* 메인 명령어가 실행 시 default option 인자는 `CMD`로 정의
* 두 명령어 모두 `[arg1, arg2, ..]` List format으로 정의



최종적으로 container orchestration에 대한 Docker-compose File을 만들면 다음과 같다.

```dockerfile
#//file: "docker-compose.yaml"
version: "3"

services:
  postgres-server:
    image: postgres:14.0
    container_name: postgres-server
    ports:
      - 5432:5432
    environment:
      POSTGRES_USER: ---
      POSTGRES_PASSWORD: ---
      POSTGRES_DB: mydatabase
      
    healthcheck:
      test: ["CMD", "pg_isready", "-q", "-U", "leepic", "-d", "mydatabase"]
      interval: 10s
      timeout: 5s
      retries: 5
  
  data-generator:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: data-generator
    depends_on:
      - postgres-server
    command: ["postgres-server"]

networks:
  default:
    name: mlops-network
```

여기서 주의깊게 봐야할 부분은 `depends_on` 인데, data-generator 컨테이너는 PostgreSQL 서버에 데이터를 INSERT하는 컨테이너임을 생각해야한다.

즉, PostgreSQL Server가 구동되어있어야하는 상태이므로, postgres-server 컨테이너를 먼저 만들고 data-generator 컨테이너를 만들어서 데이터를 INSERT 시킨다.

Health Check는 CMD 명령어로 10초마다 PostgresSQL 서버를 체크하는데, 5초를 넘어가면 retries를 1회 증가시키고 retries가 5회가 넘어가면 `exit status code`를 나타낸다.

항상. 오타에 주의하자.
{:.note}





### Appendix 1) VS code와 WSL2 연결

![image-20240908143222936](./../../../images/2024-09-08-MLOps_for_MLE_Tutorial(1)/image-20240908143222936.png)

비교적 간단하다.

`Remote Development` Extension을 설치하면 자동으로 `Remote WSL`이 설치되기 때문에 해당 Extension을 설치하고 좌측 탭에 생기는 Remote 아이콘을 클릭해서 실행 중인 WSL을 접속하면 된다.

![image-20240908143424984](./../../../images/2024-09-08-MLOps_for_MLE_Tutorial(1)/image-20240908143424984.png)



 ### Appendix 2) CONFIG 파일 관리

클라우드를 접하다보면 AWS나 GCP등 접속 Key 등, 혹은 DB나 다른 프로그램을 접속하기 위한 Key들을 code에 하드코딩하게되는데, 후에 Github에 이를 올리는 순간 AWS, GCP 같은 클라우드는 해킹을 당하거나... (그래서 요금폭탄 맞거나) local이 해킹당하거나.. 등등 위험에 처할 수 있다.

따라서 `config.py` 파일을 추가로 만들고 `.gitignore`로 무시하게 해주자..

~~~python
#//file: config.py
class DB_config:
    DB_USER="leepic"
    DB_PASSWORD="------" # write '-' for protection
    DB_DATABASE="------" # write '-' for protection
    DB_PORT=5432
~~~

git ignore를 만들어서 포함되지 않게 만들어주자



> Reference 

[Git ignore 생성](https://velog.io/@psk84/.gitignore-%EC%A0%81%EC%9A%A9%ED%95%98%EA%B8%B0)

[MLOps for MLE](https://mlops-for-mle.github.io/tutorial/docs/intro)

[ENTRYPOINT와 CMD의 차이](https://bluese05.tistory.com/77)

