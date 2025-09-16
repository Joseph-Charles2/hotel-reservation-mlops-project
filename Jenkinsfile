pipeline{
        agent any
        environment {
            VENV_DIR = 'venv_project_1'
            GCP_PROJECT = "mlops-463907"
            GCLOUD_PATH = "/var/jenkins_home/google-cloud-skd/bin"
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
            stage('Building And Pushing Docker Image To GCR'){
                steps{
                    withCredentials([file(credentialsId : 'gcp-key', variable : 'GOOGLE_APPLICATION_CREDENTIALS')]){
                         script{
                        echo 'Building And Pushing Docker Image To GCR ............'
                        sh '''
                            export PATH =${PATH}:${GCLOUD_PATH}

                            gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}

                            gcloud config set project ${GCP_PROJECT}

                            gcloud auth configure-docker --quiet

                            docker build -t gcr.io/${GCP_PROJECT}/ml-project:latest .

                            docker push gcr.io/${GCP_PROJECT}/ml-project:latest


                            '''
                    }
                    }

                 }
            }

        }
    }