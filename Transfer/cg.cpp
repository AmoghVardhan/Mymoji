#include <iostream>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <unistd.h>
#include <string>
#include <arpa/inet.h>
#include <string.h>
#include <stdio.h>

using namespace std;

#define SERVER_PORT htons(50008)

int main() {

        char buffer[1];
        int n;

        int serverSock=socket(AF_INET, SOCK_STREAM, 0);

        sockaddr_in serverAddr;
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = SERVER_PORT;
        serverAddr.sin_addr.s_addr = INADDR_ANY;

        bind(serverSock, (struct sockaddr*)&serverAddr, sizeof(struct sockaddr));


        listen(serverSock,1);

        while (1) {
                // bzero(buffer, 1000);
                sockaddr_in clientAddr;
                socklen_t sin_size=sizeof(struct sockaddr_in);
                int clientSock=accept(serverSock,(struct sockaddr*)&clientAddr, &sin_size);
                n = read(clientSock, buffer, 500);
                cout<< buffer << endl;
        }
        return 0;
}
