#include <windows.h>
#include <shlwapi.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <Tlhelp32.h>
#include <stdio.h>
#include <tchar.h>
#include <psapi.h>



using namespace std;

// 显示页面的保护方式 
void printProtection(unsigned long long dwTarget)
{
	char as[] = "----------";
	// PAGE_NOACCESS: 禁止对提交的页面区域的所有访问，读取、写入或执行提交区域会导致访问冲突异常
	// PAGE_READONLY：启动对提交的页面区域的读取访问，尝试写入提交区域会导致访问冲突
	//                如果系统区分只读访问和执行访问，则尝试执行提交区域也会导致访问冲突
	// PAGE_READWRITE：启动对提交的页面区域的读写访问
	// PAGE_WRITECOPY：为提交的页面区域提供写时复制保护
	// PAGE_EXECUTE：启动对提交的页面区域的执行访问，试图读取或写入提交的页面区域会导致访问冲突
	// PAGE_EXECUTE_READ：启动对提交的页面区域的执行和读取访问，尝试写入提交的页面区域会导致访问冲突
	// PAGE_EXECUTE_READWRITE：启动对提交的页面区域的执行、读取和写入访问
	// PAGE_EXECUTE_WRITECOPY：启动对提交的页面区域的执行、读取和写入访问，页面共享写时读和写时复制
	// PAGE_GUARD： 保护的页面保证不可访问
	// PAGE_NOCACHE：防止当其映射到虚拟页的时候被微处理器缓存
	// PAGE_WRITECOMBINE：CPU采用了合并写技术来抵消内存访问带来的延迟
	if (dwTarget & PAGE_NOACCESS) as[0] = 'N';
	if (dwTarget & PAGE_READONLY) as[1] = 'R';
	if (dwTarget & PAGE_READWRITE)as[2] = 'W';
	if (dwTarget & PAGE_WRITECOPY)as[3] = 'C';
	if (dwTarget & PAGE_EXECUTE) as[4] = 'X';
	if (dwTarget & PAGE_EXECUTE_READ) as[5] = 'r';
	if (dwTarget & PAGE_EXECUTE_READWRITE) as[6] = 'w';
	if (dwTarget & PAGE_EXECUTE_WRITECOPY) as[7] = 'c';
	if (dwTarget & PAGE_GUARD) as[8] = 'G';
	if (dwTarget & PAGE_NOCACHE) as[9] = 'D';
	if (dwTarget & PAGE_WRITECOMBINE) as[10] = 'B';
	printf("  %s  ", as);
}

// 显示系统信息 
void displaySystemConfig()
{
	// SYSTEM_INFO结构体包含了当前计算机的系统信息
	// 包括计算机的体系结构，中央处理器的类型，系统中中央处理器的数量，页面的大小以及其他信息
	SYSTEM_INFO si;
	memset(&si, 0, sizeof(si));

	// GetNativeSystemInfo 获取当前计算机的系统信息 
	GetNativeSystemInfo(&si);

    // C++支持两种字符串，常规的ANSI编码和Unicode编码
    // 微软将这两套字符集及其操作进行了统一，这就是TCHAR
    // MAX_PATH: 定义了编译器支持的最长全路径名的长度
	TCHAR str_page_size[MAX_PATH];
	// si.dwPageSize 页面大小 
	// StrFormatByteSize 将数值转换为表示大小值(以字节、千字节、兆字节或千兆字节为单位)的字符串
	StrFormatByteSize(si.dwPageSize, str_page_size, MAX_PATH);

    // si.lpMaximumApplicationAddress 应用程序和动态链接库可访问的最低虚拟内存地址信息
	// si.lpMinimumApplicationAddress 应用程序和动态链接库可访问的最高虚拟内存地址信息 
	unsigned long long memory_size = (unsigned long long)si.lpMaximumApplicationAddress - (unsigned long long)si.lpMinimumApplicationAddress;
	TCHAR str_memory_size[MAX_PATH];
	StrFormatByteSize(memory_size, str_memory_size, MAX_PATH);

    // si.wProcessorArchitecture 体系架构
    // si.dwNumberOfProcessors 处逻辑理器数量 
    // hex: 表示之后的数字以16进制方式输出
    // setw(int n)用来控制输出间隔,setw()默认填充的内容为空格,可以setfill()配合使用设置其他字符填充
	printf("系统信息如下:\n");
	cout << "------------------------------------------------" << endl;
	cout << "Processor Architecture         | " << (si.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_AMD64 || si.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_INTEL ? "x64" : "x86") << endl;
	cout << "Number Of Processors           | " << si.dwNumberOfProcessors << endl;
	cout << "Virtual Memory Page Size       | " << str_page_size << endl;
	cout << "Minimum Application Address    | 0x" << hex << setfill('0') << setw(8) << (unsigned long long)si.lpMinimumApplicationAddress << endl;
	cout << "Maximum Application Address    | 0x" << hex << setw(8) << (unsigned long long)si.lpMaximumApplicationAddress << endl;
	cout << "Total Available Virtual Memory | " << str_memory_size << endl;
	cout << "------------------------------------------------" << endl;
	return;

}

// 显示系统内存使用情况 
void displayMemoryCondition()
{

	printf("系统内存使用情况如下: \n");
	cout << "--------------------------------------------" << endl;
	// MEMORYSTATUSEX结构体包含了当前计算机可用的物理和虚拟内存信息
	// stat.dwLength 结构体的长度，在使用函数前必须初始化此值
	MEMORYSTATUSEX stat;
	stat.dwLength = sizeof(stat);
	// GlobalMemoryStatusEx 函数用来获取当前计算机可用的物理和虚拟内存信息 
	GlobalMemoryStatusEx(&stat);

	// setbase(10) 十进制输出
	// stat.dwMemoryLoad 物理内存的使用率(0至100的整数)
	// stat.ullTotalPhys 物理内存的总量,以字节为单位
	// stat.ullAvailPhys 物理内存的剩余量,以字节为单位
	// stat.ullTotalPageFile 当前系统的可提交内存量限制（以字节为单位）
	// stat.ullAvailPageFile 当前系统此时可以提交的最大内存量（以字节为单位）
	// stat.ullTotalVirtual 虚拟内存的总量
	// stat.ullAvailVirtual 虚拟内存的剩余量
	long int DIV = 1024 * 1024;
	cout << "Memory Load              | " << setbase(10) << stat.dwMemoryLoad << "%\n"
		<< "Total Physical Memory     | " << setbase(10) << stat.ullTotalPhys / DIV << "MB\n"
		<< "Available Physical Memory | " << setbase(10) << stat.ullAvailPhys / DIV << "MB\n"
		<< "Total Page File           | " << stat.ullTotalPageFile / DIV << "MB\n"
		<< "Avaliable Page File       | " << stat.ullAvailPageFile / DIV << "MB\n"
		<< "Total Virtual Memory      | " << stat.ullTotalVirtual / DIV << "MB\n"
		<< "Avaliable Virtual Memory  | " << stat.ullAvailVirtual / DIV << "MB" << endl;
	cout << "--------------------------------------------" << endl;
}

// 获取所有进程信息 
void getAllProcessInformation()
{
	printf("所有进程信息如下: \n");

	// PROCESSENTRY32 结构体存储所有进程信息 
	PROCESSENTRY32 pe32;
	// pe32.dwSize 结构体的长度，在使用函数前必须初始化此值
	pe32.dwSize = sizeof(pe32);
	// CreateToolhelp32Snapshot函数为指定的进程,进程使用的堆[HEAP],模块[MODULE],线程[THREAD]建立一个快照[snapshot]
	// TH32CS_SNAPPROCESS: 在快照中包含系统中所有进程
	// 调用成功,返回快照的句柄,调用失败,返回INVALID_HANDLE_VALUE
	HANDLE hProcessShot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
	if (hProcessShot == INVALID_HANDLE_VALUE)
	{
		printf("调用CreateToolhelp32Snapshot失败.\n");
		return;
	}

	// Process32First Process32Next 进程获取函数,hProcessShot 是快照句柄, pe32是指向PROCESSENTRY32结构的指针
	// pe32.th32ProcessID 进程ID
	// pe32.szExeFile 进程对应的可执行文件名
	cout << " |  Num  |  ProcessID  |  ProcessName" << endl;
	cout << "-----------------------------------------" << endl;
	if (Process32First(hProcessShot, &pe32)) {
		for (int i = 0; Process32Next(hProcessShot, &pe32); i++) {
			wprintf(L" | %4d  | %5d  |  %s\n", i, pe32.th32ProcessID, pe32.szExeFile);
		}
	}
	cout << "-----------------------------------------" << endl;
	CloseHandle(hProcessShot);
	return;
}

void ShowHelp()
{
	cout << "--------------------------------------------------------------------------" << endl;
	cout << "Command Type " << endl
		<< "config     : 显示系统配置" << endl
		<< "condition  : 显示系统内存使用情况" << endl
		<< "process    : 显示所有进程信息" << endl
		<< "processid  : 显示指定进程信息" << endl
		<< "exit       : 退出" << endl;
	cout << "--------------------------------------------------------------------------" << endl;
	return;
}

// 获取指定进程信息 
void getProcessDetail(int pid)
{
	// OpenProcess函数打开一个进程对象，并返回进程的句柄
	// PROCESS_ALL_ACCESS 获得所有能获得的进程访问权限
	// 0 表示所得到的进程句柄不可以被继承
	// pid 被打开进程的ID 
	HANDLE hProcess = OpenProcess(PROCESS_ALL_ACCESS, 0, pid);
	if (!hProcess) return;
	cout << " | "
		<< "   Memory Addr    | "
		<< "   Size           | "
		<< "   PageStatus     | "
		<< "   Protect        | "
		<< "   Type           | "
		<< "   ModuleName"
		<< endl;
	// 系统信息 
	SYSTEM_INFO si;					
	// ZeroMemory 用0填充一块内存区域 
	ZeroMemory(&si, sizeof(si));
	// GetSystemInfo 获取系统信息, GetNativeSystemInfo 是更低层的API 
	GetSystemInfo(&si);

	// 存储进程的页面信息
	MEMORY_BASIC_INFORMATION mbi;		 
	ZeroMemory(&mbi, sizeof(mbi));

	// si.lpMinimumApplicationAddress 应用程序和动态链接库可访问的最低虚拟内存地址信息
	// LPCVOID 无类型的指针 
	LPCVOID pBlock = (LPVOID)si.lpMinimumApplicationAddress;


	// VirtualQueryEx(hProcess, pBlock, &mbi, sizeof(mbi)查询进程的页面信息 
	// hProcess 待查询进程的句柄
	// pBlock 待查询进程页面的虚拟基地址 
	// mbi 保存相关信息的数据指针
	while (pBlock < si.lpMaximumApplicationAddress) {
		if (VirtualQueryEx(hProcess, pBlock, &mbi, sizeof(mbi)) == sizeof(mbi))
		{
			cout << " | ";
			// mbi.RegionSize 当前进程的虚拟页面大小，以字节为单位 
			LPCVOID pEnd = (PBYTE)pBlock + mbi.RegionSize;
			TCHAR szSize[MAX_PATH];
			StrFormatByteSize(mbi.RegionSize, szSize, MAX_PATH);
			// hex: 表示之后的数字以16进制方式输出
			// setw(8)用来控制输出间隔
			// 显示页面的起始地址，结束地址，页面大小 
			cout.fill('0');
			cout << hex << setw(8) << (unsigned long long)pBlock
				<< "-"
				<< hex << setw(8) << (unsigned long long)pEnd - 1
				<< " | "
				<< setw(8) << szSize;
			// 显示页面的状态 
			switch (mbi.State)
			{
				case MEM_COMMIT:cout << " | " << setw(9) << "Committed" << " | "; break;
				case MEM_FREE:cout << " | " << setw(9) << "   Free  " << " | "; break;
				case MEM_RESERVE:cout << " | " << setw(9) << " Reserved" << " | "; break;
				default: cout << "          | "; break;
			}
			// 显示页面的保护方式 
			if (mbi.Protect == 0 && mbi.State != MEM_FREE)
			{

				mbi.Protect = PAGE_READONLY;

			}
			printProtection(mbi.Protect);
			// 显示页面类型
			switch (mbi.Type)
			{
				case MEM_IMAGE:cout << " |  Image  | "; break;
				case MEM_PRIVATE:cout << " | Private | "; break;
				case MEM_MAPPED:cout << " |  Mapped | "; break;
				default:cout << " |         | "; break;
			} 
			TCHAR str_module_name[MAX_PATH];
			// GetModuleFileName((HMODULE)pBlock, str_module_name, MAX_PATH)获取可执行文件的绝对路径
			// pBlock 可执行文件指针，NULL则指向当前程序文件 
			// PathStripPath 删除路径名 
			if (GetModuleFileName((HMODULE)pBlock, str_module_name, MAX_PATH) > 0) {
				PathStripPath(str_module_name);
				wprintf(L"%s", str_module_name);
			}
			cout << endl;
			// 移动到下一个虚拟页面 
			pBlock = pEnd;	
		}
	}
}

int main()
{
	// Setlocale()是一个系统函数，功能是用来配置地域的信息，设置当前程序使用本地化信息
	setlocale(LC_ALL, "CHS");
	cout << endl << "*----------内存监视器----------*" << endl << endl;
	cout << "--Type 'help' for help.\n" << endl;
	string cmd;
	char cmd_charstr[127];
	while (1)
	{
		cout << "Manager>";
		cin.getline(cmd_charstr, 127);
		cmd = cmd_charstr;

		if (cmd == "config") {
			cout << endl;
			displaySystemConfig();
		}
		else if (cmd == "condition") {
			cout << endl;
			displayMemoryCondition();
		}
		else if (cmd == "process") {
			cout << endl;
			getAllProcessInformation();
		}
		else if (cmd == "processid") {
			cout << "ProcessID> ";
			int pid = 0;
			cin >> pid;
			cin.getline(cmd_charstr, 127);
			if (pid <= 0) continue;
			cout << endl;
			getProcessDetail(pid);
		}
		else if (cmd == "help") {
			cout << endl;
			ShowHelp();
		}
		else if (cmd == "exit") {
			break;
		}
		else if (cmd == "clear" || cmd == "cls") {
			// system("cls") 清屏函数 
			system("cls");
		}
		else {
			if (cmd != "") cout << "Invalid command, maybe you can type \"help\"?." << endl;
			// fflush(stdin) 清空输入缓冲流 
			fflush(stdin);
			// cin.clear()清理错误表示符
			cin.clear();
			continue;
		}
		cin.clear();

	}

	system("pause");
	return 0;

}
