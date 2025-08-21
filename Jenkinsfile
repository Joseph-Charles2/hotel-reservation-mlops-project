pipeline{
        agent any

        stages{
            stage('cloning Github repo to Jenkins'){
                steps{
                    script{
                        echo 'cloning Github repo to Jenkins ............'
                        checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[url: 'https://github.com/Joseph-Charles2/hotel-reservation-mlops-project.git']])
                    }
              }  }
        }
    }