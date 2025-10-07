pipeline
{
    agent any

    stages
    {
        stage('Cloning Github repo to jenkins')
        {
            steps{
                script{
                    echo 'Cloning Github repo to Jenkins'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-tokens', url: 'https://github.com/Joseph-Charles2/hotel-reservation-mlops-project.git']])

                }
            }
        }
    }
}