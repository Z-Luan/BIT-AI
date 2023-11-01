#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<windows.h>

#define Buffer_num 6
#define Buffer_len 10
#define Producer_num 2
#define Producer_repeat 12
#define Consumer_num 3
#define Consumer_repeat 8
#define Process_num 5

LARGE_INTEGER Start_Time, End_Time, Frequency;
HANDLE semaphore_empty, semaphore_fill, semaphore_mutex;
HANDLE Process_Handle[Process_num + 1];

struct shared_memory
{
    char Buffer[Buffer_num][Buffer_len + 1];
    int BEGIN;
    int END;
};

HANDLE MakeSharedFile()
{
    // 每个进程会被分配一块独立的内存空间, 不能由其他进程访问, 通过在进程间共享内存映像文件可以达到共享内存的目的
    // CreateFileMapping() 把文件映射到内存
    // INVALID_HANDLE_VALUE: 表示在页面文件(虚拟内存)中创建一个可共享的文件映射 
    // NULL: 使用默认安全配置
    // PAGE_READWRITE: 以可读可写的方式打开文件映射
    // 0: 文件映射大小的高32位
    // sizeof(struct shared_memory): 文件映射大小的低32位
    // 由于 Windows 支持的最大文件大小可以用 64 位整数表示, 因此必须使用两个 32 位值, 对于小于 4GB 的文件来说, dwMaximumSizeHigh 为 0
    // "SHARED_MEMORY": 共享内存名称
	HANDLE FileMapping = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(struct shared_memory), "SHARED_MEMORY");
	return FileMapping;
}

void Create_Semaphore()
{
    // CreateSemaphore(lpSemaphoreAttributes, lInitialCount, lMaximumCount, lpName)创建一个信号量对象
    // lpSemaphoreAttributes: 信号量的属性, 一般可以设置为NULL
    // lInitialCount: 信号量初始值, 必须大于等于0, 而且小于等于lMaximumCount
    // lMaximumCount: 信号量的最大值, 必须大于0
    // lpName: 信号量的名字, 可设置为NULL, 表示无名的信号量
    semaphore_empty = CreateSemaphore(NULL, Buffer_num, Buffer_num, "SEMAPHORE_EMPTY");
    semaphore_fill = CreateSemaphore(NULL, 0, Buffer_num, "SEMAPHORE_FILL");
    semaphore_mutex = CreateSemaphore(NULL, 1, 1, "SEMAPHORE_MUTEX");
}

void Create_Process(int Process_ID)
{
    // 当Windows创建新进程时, 将使用STARTUPINFO结构体的成员
    STARTUPINFO S;
    // STARTUPINFO结构体的所有成员初始化为 0 
    memset(&S, 0, sizeof(S));
    // 成员 cb 包含STARTUPINFO结构体的字节数, 必须进行初始化
    S.cb = sizeof(S);

    // typedef struct _PROCESS_INFORMATION { 
    //     HANDLE hProcess;   //存放每个对象的与进程相关的句柄 
    //     HANDLE hThread;    //返回的线程句柄 
    //     DWORD dwProcessId; //用来存放进程ID号 
    //     DWORD dwThreadId;  //用来存放线程ID号 
    // } PROCESS_INFORMATION, *PPROCESS_INFORMATION, *LPPROCESS_INFORMATION;
    PROCESS_INFORMATION P;

    char Cmd_Str[105];
    char File_Path[100];

    // GetModuleFileName(NULL,buff,MAX_PATH)获取exe可执行文件的绝对路径
    // NULL: 可执行文件为当前程序
    // buff: 存放地址的指针
    GetModuleFileName(NULL, File_Path, sizeof(File_Path));

    // sprintf(char *str,const char *format,...) 发送格式化输出到 str 所指向的字符串
    sprintf(Cmd_Str, "%s %d", File_Path, Process_ID);

    // BOOL CreateProcess(  
    // LPCTSTR lpApplicationName, // 应用程序名称, 指向启动进程的可执行文件, NULL表示可执行文件为当前程序
    // LPTSTR lpCommandLine, // 命令行字符串, 启动进程的命令行中的参数  
    // LPSECURITY_ATTRIBUTES lpProcessAttributes, // 进程的安全属性, NULL表示默认安全属性
    // LPSECURITY_ATTRIBUTES lpThreadAttributes, // 线程的安全属性, NULL表示默认安全属性
    // BOOL bInheritHandles, // 是否继承父进程的属性  
    // DWORD dwCreationFlags, // 创建标志, 表示进程的创建标志以及优先级控制  
    // LPVOID lpEnvironment, // 指向新进程的环境变量块, 如果设置为 NULL, 那么使用父进程的环境变量  
    // LPCTSTR lpCurrentDirectory, // 指定创建后新进程的当前目录, 如果设置为 NULL, 那么就在父进程所在的当前目录  
    // LPSTARTUPINFO lpStartupInfo, // 传递给新进程的信息, 指向一个 STARTUPINFO 结构, 该结构里可以设定启动信息, 可以设置为 NULL  
    // LPPROCESS_INFORMATION lpProcessInformation // 新进程返回的信息, 指向一个 PROCESS_INFORMATION 结构, 返回被创建进程的信息  
    // ); 
    CreateProcess(NULL, Cmd_Str, NULL, NULL, FALSE, 0, NULL, NULL, &S, &P);
    Process_Handle[Process_ID] = P.hProcess;

    return;
}

void Close_Semaphore()
{
    CloseHandle(semaphore_empty);
    CloseHandle(semaphore_fill);
    CloseHandle(semaphore_mutex);
}

void Open_Semaphore()
{
    // OpenSemaphore()通过信号量名, 获得信号量对象句柄
    // 第一个参数表示访问权限, 一般传入SEMAPHORE_ALL_ACCESS
    // 第二个参数表示信号量句柄继承性
    // 第三个参数表示信号量名称，不同进程可以通过信号量名称名称来确保它们访问同一个信号量
    semaphore_empty = OpenSemaphore(SEMAPHORE_ALL_ACCESS,FALSE, "SEMAPHORE_EMPTY");
    semaphore_fill = OpenSemaphore(SEMAPHORE_ALL_ACCESS,FALSE, "SEMAPHORE_FILL");
    semaphore_mutex = OpenSemaphore(SEMAPHORE_ALL_ACCESS,FALSE, "SEMAPHORE_MUTEX");
}

void Show_Buffer(struct shared_memory *sm)
{
    printf("缓冲池映像为: ");
    for (int i = 0; i < Buffer_num; i++){
        printf("|%-15s", sm->Buffer[i]);
        // printf("%d",sizeof(sm->Buffer[i]));
    }   
    printf("\n");
}

char* Create_Sentence()
{
    static char sentence[Buffer_len];
    memset(sentence, 0, sizeof(sentence));
    int num = rand() % 10 + 1;
    for (int i = 0; i < num; i++)
        sentence[i] = (char)(rand() % 26 + 65);
    return sentence;
}


void Producer(int Process_ID)
{
    // OpenFileMapping 打开文件映射对象, 返回指定文件映射对象的 HANDLE
    // FILE_MAP_ALL_ACCESS: 指定对文件映射的访问方式, 需要与CreateFileMapping()中设置的保护属性相匹配
    // FALSE: 函数返回的 HANDLE 不能由当前进程启动的新进程继承
    // "SHARED_MEMORY": 文件映射对象名称

    // LPVOID 是一个没有类型的指针
    // MapViewOfFile(HANDLE hFileMappingObject, DWORD dwDesiredAccess, DWORD dwFileOffsetHigh, DWORD dwFileOffsetLow, DWORD dwNumberOfBytesToMap)把内存中的文件映射到进程的地址空间中
    // FileMapping: CreateFileMapping()返回的文件映射对象 HANDLE
    // FILE_MAP_ALL_ACCESS: 指定对文件映射的访问方式, 需要与CreateFileMapping()中设置的保护属性相匹配
    // MapViewOfFile()函数允许映射全部或部分文件, 在映射时需要指定文件的偏移地址以及待映射的长度
    // 文件的偏移地址由参数 dwFileOffsetHigh 和 dwFileOffsetLow 组成的64位值来确定
    // dwNumberOfBytesToMap 为0表示映射整个文件

    // ZeroMemory(PVOID Destination, SIZE_T Length) 用0来填充一块内存区域
    // PVOID Destination: 指向一块准备用0来填充的内存区域的起始地址
    // Length: 准备用0来填充的内存区域的大小，按字节来计算
	// ZeroMemory(File, sizeof(struct shared_memory))
    HANDLE FileMapping = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, "SHARED_MEMORY");
    LPVOID File = MapViewOfFile(FileMapping, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    struct shared_memory *sm = (struct shared_memory*)(File);

    Open_Semaphore();

    for(int i = 0 ; i < Producer_repeat ; i++)
    {
        QueryPerformanceCounter(&Start_Time);

        // WaitForSingleObject() 等待信号量 >= 0, 执行 P 操作
        // semaphore_empty , semaphore_mutex等待信号量的句柄
        // INFINITE: 允许无限等待
        WaitForSingleObject(semaphore_empty, INFINITE);
        WaitForSingleObject(semaphore_mutex, INFINITE);

        srand((unsigned int)time(NULL));
        int Time = (rand() % 5 + 1) * 500; 
        // Sleep 的单位是毫秒
        Sleep(Time);

        char* sentence = sm->Buffer[sm->END];
        char* create_sentence = Create_Sentence();
        strncpy(sentence, create_sentence, Buffer_len);
        printf("进程%d: 生产者在%d号缓冲区添加  %s\n", Process_ID, sm->END, sm->Buffer[sm->END]);
        sm->END = (sm->END + 1) % Buffer_num;
  
        QueryPerformanceCounter(&End_Time);
        QueryPerformanceFrequency(&Frequency);
        // QueryPerformanceFrequency 获取机器内部定时器的时钟频率, 以 周期变化次数/微秒 为单位
        double Running_Time = (double)(End_Time.QuadPart - Start_Time.QuadPart) * 1000 / Frequency.QuadPart;
        printf("时间为: ");
        printf("%lf ms\n", Running_Time);

        Show_Buffer(sm);

        // BOOL ReleaseSemaphore(
        //      HANDLE hSemaphore, //信号量对象的句柄
        //      LONG   lReleaseCount, //信号量对象当前计数将增加的量
        //      LPLONG lpPreviousCount //用于接收信号量的上一个计数, 如果不需要可设置为NULL
        // );
        ReleaseSemaphore(semaphore_fill, 1, NULL); // 执行 v 操作
        ReleaseSemaphore(semaphore_mutex, 1, NULL);
    }

    Close_Semaphore();

    // 在完成对映射到进程地址空间区域的文件处理后，需要通过函数UnmapViewOfFile()完成对文件映像的释放
    // File: MapViewOfFile()的返回值
    UnmapViewOfFile(File);
    CloseHandle(FileMapping);
}

void Consumer(int Process_ID)
{
    HANDLE FileMapping = OpenFileMapping(FILE_MAP_ALL_ACCESS,FALSE,"SHARED_MEMORY");
    LPVOID File = MapViewOfFile(FileMapping, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    struct shared_memory *sm = (struct shared_memory*)(File);

    Open_Semaphore();

    for(int i = 0 ; i < Consumer_repeat ; i++)
    {
        QueryPerformanceCounter(&Start_Time);

        WaitForSingleObject(semaphore_fill, INFINITE);
        WaitForSingleObject(semaphore_mutex, INFINITE);

        srand((unsigned int)time(NULL));
        int Time = (rand() % 5 + 1) * 1000;
        Sleep(Time);

        char* sentence = sm->Buffer[sm->BEGIN];
        printf("进程%d: 消费者在%d号缓冲区读取  %s\n", Process_ID, sm->BEGIN, sm->Buffer[sm->BEGIN]);
        memset(sentence, 0, sizeof(sm->Buffer[sm->BEGIN]));
        sm->BEGIN = (sm->BEGIN + 1) % Buffer_num;

        QueryPerformanceCounter(&End_Time);
        QueryPerformanceFrequency(&Frequency);
        double Running_Time = (double)(End_Time.QuadPart - Start_Time.QuadPart) * 1000 / Frequency.QuadPart;
        printf("时间为: ");
        printf("%lf ms\n", Running_Time);

        Show_Buffer(sm);

        ReleaseSemaphore(semaphore_empty, 1, NULL);
        ReleaseSemaphore(semaphore_mutex, 1, NULL);
    }

    Close_Semaphore();
    UnmapViewOfFile(File);
    CloseHandle(FileMapping);
}

// argc是命令行总的参数个数, 默认值为1
// argv[]包含argc个参数, 其中argv[0]是程序的全名，其余参数为命令行后面跟着的用户输入参数
int main(int argc, char *argv[])
{
    if(argc==1)
    {
        // HANDLE, 本质上是一个 Long 型数据, 是一种指向指针的指针

        // 为什么要设置HANDLE, 直接用指针调用对象不可以吗?
        // Windows是一个以虚拟内存为基础的操作系统, 在这种系统环境下, Windows内存管理器经常在内存中来回移动对象, 依此来满足各种应用程序的内存需求
        // 对象被移动意味着对象的地址变化了, 为了解决这个问题, Windows操作系统为各应用程序腾出一块内存, 用来专门登记各应用程序对象在内存中的地址变化
        // 而各应用程序在外存中的存储位置是不变的, Windows内存管理器在移动对象到内存中后, 把对象的新地址告知 HANDLE 来保存
        // 这样我们只需记住 HANDLE 地址就可以间接地知道各个程序对象具体存储在内存中的哪个位置
        HANDLE FileMapping = MakeSharedFile();

        // 创建信号量
        Create_Semaphore();

        printf("进程1,2为生产者进程，进程3,4,5为消费者进程\n");
        for(int i = 1 ; i <= Process_num ; i++)
        {
            Create_Process(i);
        }

        // DWORD WaitForMultipleObjects(  // 每个word为2个字节的长度, DWORD双字即为4个字节, 每个字节是8位二进制
        // DWORD nCount,             // 句柄数量  
        // CONST HANDLE *lpHandles,  // 句柄数组的指针  
        // BOOL fWaitAll,            // 等待类型, 如果为TRUE, 表示除非对象都发出信号, 否则就一直等待下去, 如果FALSE, 表示任何对象发出信号即可   
        // DWORD dwMilliseconds      // 指定要等候的毫秒数, 如果设置为零, 表示立即返回, 如指定常数INFINITE, 则根据实际情况无限等待下去 
        // ); 
        WaitForMultipleObjects(Process_num, Process_Handle + 1, TRUE, INFINITE);

        // CloseHandle() 关闭句柄对象
        for(int i = 1 ; i <= Process_num ; i++)
        {
            CloseHandle(Process_Handle[i]);
        }

        Close_Semaphore();
        CloseHandle(FileMapping);
        printf("运行完成\n");
    }
    else
    {   // atoi() 将字符串类型强制转化为整型
        int Process_ID = atoi(argv[1]);
        if(Process_ID <= Producer_num)
            Producer(Process_ID);
        else
            Consumer(Process_ID);
    }

    return 0;
}
