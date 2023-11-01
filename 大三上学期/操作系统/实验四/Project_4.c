#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <dirent.h>
#include <utime.h>

const int buffer_size=1024;

void Change_Attr(char *file_sor,char *file_tar)
{
    struct stat statbuf;
    lstat(file_sor,&statbuf);
    chmod(file_tar,statbuf.st_mode);
    chown(file_tar,statbuf.st_uid,statbuf.st_gid);
    if(S_ISLNK(statbuf.st_mode))
    {
        struct timeval time_sor[2];
        time_sor[0].tv_sec=statbuf.st_mtime;
        time_sor[1].tv_sec=statbuf.st_ctime;
        lutimes(file_tar,time_sor);
    }
    else
    {
        struct utimbuf utime_sor;
        utime_sor.actime=statbuf.st_atime;
        utime_sor.modtime=statbuf.st_mtime;
        utime(file_tar,&utime_sor);
    }
}

void Copy_File(char *file_sor,char *file_tar)
{
    int fd_sor=open(file_sor,O_RDONLY);
    int fd_tar=creat(file_tar,O_WRONLY);
    unsigned char buff[buffer_size];
    int len;
    while((len=read(fd_sor,buff,buffer_size))>0)
    {
        write(fd_tar,buff,len);
    }
    Change_Attr(file_sor,file_tar);

    close(fd_sor);
    close(fd_tar);

}

void Copy_Link(char *lnk_sor,char *lnk_tar)
{
    unsigned char path[buffer_size];
    readlink(lnk_sor,path,buffer_size);
    symlink(path,lnk_tar);
    Change_Attr(lnk_sor,lnk_tar);
}

void Copy_Dir(char *dir_sor,char *dir_tar)
{
    DIR *pd_sor=opendir(dir_sor);
    struct dirent *entry_sor=NULL;
    struct stat statbuf;
    while((entry_sor=readdir(pd_sor)))
    {
        if(strcmp(entry_sor->d_name,".")==0||
           strcmp(entry_sor->d_name,"..")==0)
            continue;
        
        char ts[buffer_size],tt[buffer_size];
        strcpy(ts,dir_sor);
        strcat(ts,"/");
        strcat(ts,entry_sor->d_name);
        strcpy(tt,dir_tar);
        strcat(tt,"/");
        strcat(tt,entry_sor->d_name);
        lstat(ts,&statbuf);
        if(S_ISLNK(statbuf.st_mode))
        {
            Copy_Link(ts,tt);
        }
        else if(S_ISDIR(statbuf.st_mode))
        {
            
            mkdir(tt,statbuf.st_mode);
            Copy_Dir(ts,tt);
        }
        else if(S_ISREG(statbuf.st_mode))
        {
            Copy_File(ts,tt);
        }
    }
    Change_Attr(dir_sor,dir_tar);
}

int main(int argc,char *argv[])
{
    if(argc!=3)
    {
        printf("error\n");
        return 0;
    }
    struct stat statbuf;
    lstat(argv[1],&statbuf);

    if(S_ISDIR(statbuf.st_mode))
    {
        if(opendir(argv[1])==NULL)
        {
            printf("can't find source\n");
            return 0;
        }
        if(opendir(argv[2])==NULL)
        {
            mkdir(argv[2],statbuf.st_mode);
        }
        Copy_Dir(argv[1],argv[2]);
    }

}