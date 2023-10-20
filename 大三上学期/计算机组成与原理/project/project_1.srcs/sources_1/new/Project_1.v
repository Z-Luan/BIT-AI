`timescale 1ns / 1ps

module Project_1(clk,in1,in2,WR,result);

input[31:0] in1;// 输入被除数
input[31:0] in2;// 输入除数
input clk;// 时钟信号
input WR;// WR = 0读取数据, WR = 1计算数据
reg flag;// 是否已经读取数据
reg[32:0] reg_A;// 存储被除数和余数, 双符号位, 31位数值位
reg[32:0] reg_B;// 存储除数, 双符号位, 31位数值位
reg[32:0] reg_Bfan;//存储除数的变补, 双符号位, 31位数值位
reg[31:0] reg_C;// 存储商, 单符号位，31位数值位
reg[5:0] i;//循环计数
output reg[31:0] result;// 商

// 在时钟信号 clk 上升沿时进入过程块, 根据 WR 判断是要读取数据，还是计算数据
always@(posedge clk)
begin
	if(WR == 0)
	begin
		 reg_A = {in1[31],in1[31:0]}; //双符号位
		 reg_B = {in2[31],in2[31:0]}; 
		 reg_Bfan = ~reg_B+1;//变补，连同符号位取反后，末位加1 
		 reg_C = 32'b0;
		 flag = 1; // 已经读取数据
	end
	else if(flag == 1)// 判断寄存器中是否已经读入数据，若已经读入数据，则进行除法运算
	begin
		if(reg_A[32] == reg_B[32])//被除数与除数符号相同
			begin reg_A = reg_A + reg_Bfan; end // 如果被除数与除数符号相同，则相减判断是否"够减"
		else
			begin reg_A = reg_A + reg_B; end//如果被除数与除数符号位不同，则相加判断是否"够减"
		
		for(i = 1 ; i <= 31 ; i = i + 1)//参与运算的被除数与除数数值位为n位，则左移n次，累加n+1次
		begin
			if(reg_A[32] == reg_B[32])//如果部分余数与除数符号相同，则上商1
				begin
				    reg_C[0] = 1;
				    reg_C = {reg_C[30:0],1'b0};
				    reg_A = {reg_A[32],reg_A[30:0],1'b0}; // 双符号位补码移位: 真符位不变, 算术移位
				    reg_A = reg_A + reg_Bfan;//如果部分余数与除数符号相同，则部分余数左移，然后与除数相减
				end 
			else//如果部分余数与除数符号不同，则上商0
				begin
				    reg_C[0] = 0;
				    reg_C = {reg_C[30:0],1'b0}; 
				    reg_A = {reg_A[32],reg_A[30:0],1'b0}; // 双符号位补码移位: 真符位不变, 算术移位
				    reg_A = reg_A + reg_B;//如果部分余数与除数符号不同，则部分余数左移，然后与除数相加
				end
		end
		reg_C[0] = 1;// 末尾置1
		result = reg_C;//输出结果
		flag=0;
	end
end
endmodule