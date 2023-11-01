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

// ��ʾҳ��ı�����ʽ 
void printProtection(unsigned long long dwTarget)
{
	char as[] = "----------";
	// PAGE_NOACCESS: ��ֹ���ύ��ҳ����������з��ʣ���ȡ��д���ִ���ύ����ᵼ�·��ʳ�ͻ�쳣
	// PAGE_READONLY���������ύ��ҳ������Ķ�ȡ���ʣ�����д���ύ����ᵼ�·��ʳ�ͻ
	//                ���ϵͳ����ֻ�����ʺ�ִ�з��ʣ�����ִ���ύ����Ҳ�ᵼ�·��ʳ�ͻ
	// PAGE_READWRITE���������ύ��ҳ������Ķ�д����
	// PAGE_WRITECOPY��Ϊ�ύ��ҳ�������ṩдʱ���Ʊ���
	// PAGE_EXECUTE���������ύ��ҳ�������ִ�з��ʣ���ͼ��ȡ��д���ύ��ҳ������ᵼ�·��ʳ�ͻ
	// PAGE_EXECUTE_READ���������ύ��ҳ�������ִ�кͶ�ȡ���ʣ�����д���ύ��ҳ������ᵼ�·��ʳ�ͻ
	// PAGE_EXECUTE_READWRITE���������ύ��ҳ�������ִ�С���ȡ��д�����
	// PAGE_EXECUTE_WRITECOPY���������ύ��ҳ�������ִ�С���ȡ��д����ʣ�ҳ�湲��дʱ����дʱ����
	// PAGE_GUARD�� ������ҳ�汣֤���ɷ���
	// PAGE_NOCACHE����ֹ����ӳ�䵽����ҳ��ʱ��΢����������
	// PAGE_WRITECOMBINE��CPU�����˺ϲ�д�����������ڴ���ʴ������ӳ�
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

// ��ʾϵͳ��Ϣ 
void displaySystemConfig()
{
	// SYSTEM_INFO�ṹ������˵�ǰ�������ϵͳ��Ϣ
	// �������������ϵ�ṹ�����봦���������ͣ�ϵͳ�����봦������������ҳ��Ĵ�С�Լ�������Ϣ
	SYSTEM_INFO si;
	memset(&si, 0, sizeof(si));

	// GetNativeSystemInfo ��ȡ��ǰ�������ϵͳ��Ϣ 
	GetNativeSystemInfo(&si);

    // C++֧�������ַ����������ANSI�����Unicode����
    // ΢���������ַ����������������ͳһ�������TCHAR
    // MAX_PATH: �����˱�����֧�ֵ��ȫ·�����ĳ���
	TCHAR str_page_size[MAX_PATH];
	// si.dwPageSize ҳ���С 
	// StrFormatByteSize ����ֵת��Ϊ��ʾ��Сֵ(���ֽڡ�ǧ�ֽڡ����ֽڻ�ǧ���ֽ�Ϊ��λ)���ַ���
	StrFormatByteSize(si.dwPageSize, str_page_size, MAX_PATH);

    // si.lpMaximumApplicationAddress Ӧ�ó���Ͷ�̬���ӿ�ɷ��ʵ���������ڴ��ַ��Ϣ
	// si.lpMinimumApplicationAddress Ӧ�ó���Ͷ�̬���ӿ�ɷ��ʵ���������ڴ��ַ��Ϣ 
	unsigned long long memory_size = (unsigned long long)si.lpMaximumApplicationAddress - (unsigned long long)si.lpMinimumApplicationAddress;
	TCHAR str_memory_size[MAX_PATH];
	StrFormatByteSize(memory_size, str_memory_size, MAX_PATH);

    // si.wProcessorArchitecture ��ϵ�ܹ�
    // si.dwNumberOfProcessors ���߼��������� 
    // hex: ��ʾ֮���������16���Ʒ�ʽ���
    // setw(int n)��������������,setw()Ĭ����������Ϊ�ո�,����setfill()���ʹ�����������ַ����
	printf("ϵͳ��Ϣ����:\n");
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

// ��ʾϵͳ�ڴ�ʹ����� 
void displayMemoryCondition()
{

	printf("ϵͳ�ڴ�ʹ���������: \n");
	cout << "--------------------------------------------" << endl;
	// MEMORYSTATUSEX�ṹ������˵�ǰ��������õ�����������ڴ���Ϣ
	// stat.dwLength �ṹ��ĳ��ȣ���ʹ�ú���ǰ�����ʼ����ֵ
	MEMORYSTATUSEX stat;
	stat.dwLength = sizeof(stat);
	// GlobalMemoryStatusEx ����������ȡ��ǰ��������õ�����������ڴ���Ϣ 
	GlobalMemoryStatusEx(&stat);

	// setbase(10) ʮ�������
	// stat.dwMemoryLoad �����ڴ��ʹ����(0��100������)
	// stat.ullTotalPhys �����ڴ������,���ֽ�Ϊ��λ
	// stat.ullAvailPhys �����ڴ��ʣ����,���ֽ�Ϊ��λ
	// stat.ullTotalPageFile ��ǰϵͳ�Ŀ��ύ�ڴ������ƣ����ֽ�Ϊ��λ��
	// stat.ullAvailPageFile ��ǰϵͳ��ʱ�����ύ������ڴ��������ֽ�Ϊ��λ��
	// stat.ullTotalVirtual �����ڴ������
	// stat.ullAvailVirtual �����ڴ��ʣ����
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

// ��ȡ���н�����Ϣ 
void getAllProcessInformation()
{
	printf("���н�����Ϣ����: \n");

	// PROCESSENTRY32 �ṹ��洢���н�����Ϣ 
	PROCESSENTRY32 pe32;
	// pe32.dwSize �ṹ��ĳ��ȣ���ʹ�ú���ǰ�����ʼ����ֵ
	pe32.dwSize = sizeof(pe32);
	// CreateToolhelp32Snapshot����Ϊָ���Ľ���,����ʹ�õĶ�[HEAP],ģ��[MODULE],�߳�[THREAD]����һ������[snapshot]
	// TH32CS_SNAPPROCESS: �ڿ����а���ϵͳ�����н���
	// ���óɹ�,���ؿ��յľ��,����ʧ��,����INVALID_HANDLE_VALUE
	HANDLE hProcessShot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
	if (hProcessShot == INVALID_HANDLE_VALUE)
	{
		printf("����CreateToolhelp32Snapshotʧ��.\n");
		return;
	}

	// Process32First Process32Next ���̻�ȡ����,hProcessShot �ǿ��վ��, pe32��ָ��PROCESSENTRY32�ṹ��ָ��
	// pe32.th32ProcessID ����ID
	// pe32.szExeFile ���̶�Ӧ�Ŀ�ִ���ļ���
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
		<< "config     : ��ʾϵͳ����" << endl
		<< "condition  : ��ʾϵͳ�ڴ�ʹ�����" << endl
		<< "process    : ��ʾ���н�����Ϣ" << endl
		<< "processid  : ��ʾָ��������Ϣ" << endl
		<< "exit       : �˳�" << endl;
	cout << "--------------------------------------------------------------------------" << endl;
	return;
}

// ��ȡָ��������Ϣ 
void getProcessDetail(int pid)
{
	// OpenProcess������һ�����̶��󣬲����ؽ��̵ľ��
	// PROCESS_ALL_ACCESS ��������ܻ�õĽ��̷���Ȩ��
	// 0 ��ʾ���õ��Ľ��̾�������Ա��̳�
	// pid ���򿪽��̵�ID 
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
	// ϵͳ��Ϣ 
	SYSTEM_INFO si;					
	// ZeroMemory ��0���һ���ڴ����� 
	ZeroMemory(&si, sizeof(si));
	// GetSystemInfo ��ȡϵͳ��Ϣ, GetNativeSystemInfo �Ǹ��Ͳ��API 
	GetSystemInfo(&si);

	// �洢���̵�ҳ����Ϣ
	MEMORY_BASIC_INFORMATION mbi;		 
	ZeroMemory(&mbi, sizeof(mbi));

	// si.lpMinimumApplicationAddress Ӧ�ó���Ͷ�̬���ӿ�ɷ��ʵ���������ڴ��ַ��Ϣ
	// LPCVOID �����͵�ָ�� 
	LPCVOID pBlock = (LPVOID)si.lpMinimumApplicationAddress;


	// VirtualQueryEx(hProcess, pBlock, &mbi, sizeof(mbi)��ѯ���̵�ҳ����Ϣ 
	// hProcess ����ѯ���̵ľ��
	// pBlock ����ѯ����ҳ����������ַ 
	// mbi ���������Ϣ������ָ��
	while (pBlock < si.lpMaximumApplicationAddress) {
		if (VirtualQueryEx(hProcess, pBlock, &mbi, sizeof(mbi)) == sizeof(mbi))
		{
			cout << " | ";
			// mbi.RegionSize ��ǰ���̵�����ҳ���С�����ֽ�Ϊ��λ 
			LPCVOID pEnd = (PBYTE)pBlock + mbi.RegionSize;
			TCHAR szSize[MAX_PATH];
			StrFormatByteSize(mbi.RegionSize, szSize, MAX_PATH);
			// hex: ��ʾ֮���������16���Ʒ�ʽ���
			// setw(8)��������������
			// ��ʾҳ�����ʼ��ַ��������ַ��ҳ���С 
			cout.fill('0');
			cout << hex << setw(8) << (unsigned long long)pBlock
				<< "-"
				<< hex << setw(8) << (unsigned long long)pEnd - 1
				<< " | "
				<< setw(8) << szSize;
			// ��ʾҳ���״̬ 
			switch (mbi.State)
			{
				case MEM_COMMIT:cout << " | " << setw(9) << "Committed" << " | "; break;
				case MEM_FREE:cout << " | " << setw(9) << "   Free  " << " | "; break;
				case MEM_RESERVE:cout << " | " << setw(9) << " Reserved" << " | "; break;
				default: cout << "          | "; break;
			}
			// ��ʾҳ��ı�����ʽ 
			if (mbi.Protect == 0 && mbi.State != MEM_FREE)
			{

				mbi.Protect = PAGE_READONLY;

			}
			printProtection(mbi.Protect);
			// ��ʾҳ������
			switch (mbi.Type)
			{
				case MEM_IMAGE:cout << " |  Image  | "; break;
				case MEM_PRIVATE:cout << " | Private | "; break;
				case MEM_MAPPED:cout << " |  Mapped | "; break;
				default:cout << " |         | "; break;
			} 
			TCHAR str_module_name[MAX_PATH];
			// GetModuleFileName((HMODULE)pBlock, str_module_name, MAX_PATH)��ȡ��ִ���ļ��ľ���·��
			// pBlock ��ִ���ļ�ָ�룬NULL��ָ��ǰ�����ļ� 
			// PathStripPath ɾ��·���� 
			if (GetModuleFileName((HMODULE)pBlock, str_module_name, MAX_PATH) > 0) {
				PathStripPath(str_module_name);
				wprintf(L"%s", str_module_name);
			}
			cout << endl;
			// �ƶ�����һ������ҳ�� 
			pBlock = pEnd;	
		}
	}
}

int main()
{
	// Setlocale()��һ��ϵͳ�������������������õ������Ϣ�����õ�ǰ����ʹ�ñ��ػ���Ϣ
	setlocale(LC_ALL, "CHS");
	cout << endl << "*----------�ڴ������----------*" << endl << endl;
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
			// system("cls") �������� 
			system("cls");
		}
		else {
			if (cmd != "") cout << "Invalid command, maybe you can type \"help\"?." << endl;
			// fflush(stdin) ������뻺���� 
			fflush(stdin);
			// cin.clear()��������ʾ��
			cin.clear();
			continue;
		}
		cin.clear();

	}

	system("pause");
	return 0;

}
