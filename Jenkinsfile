pipeline{
        agent any
        environment {
            VENV_DIR = 'venv_project_1'
        }
        stages{
            stage('cloning Github repo to Jenkins'){
                steps{
                    script{
                        echo 'cloning Github repo to Jenkins ............'
                        checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[url: 'https://github.com/Joseph-Charles2/hotel-reservation-mlops-project.git']])
                    }
                 }
            }
            stage('Setting up our Virtual Environment and Installing Dependencies'){
                steps{
                    script{
                        echo 'Setting up our Virtual Environment and Installing Dependencies ............'
                        sh '''
                            python -m venv ${VENV_DIR}
                            . ${VENV_DIR}/bin/activate
                            pip install --upgrade pip
                            pip install -e .
                            '''
                    }
                 }
            }

        }
    }