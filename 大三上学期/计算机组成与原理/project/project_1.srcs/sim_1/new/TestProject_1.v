`timescale 1ns / 1ps

module TestProject_1(
    );
    reg[31:0] in1, in2;//输入32位补码被除数与32位补码除数
    reg WR, clk;//WR = 0读取数据, WR = 1计算数据, clk 时钟信号
    wire [31:0] result;//商
    
    initial begin
        clk = 1'b0;
        #5 clk = 1'b1;//5s
        #5 clk = 1'b0;
        #5 clk = 1'b1;//15s
//        #5 clk = 1'b0;
//        #5 clk = 1'b1;//25s
//        #5 clk = 1'b0;
//        #5 clk = 1'b1;//35s
//        #5 clk = 1'b0;
//        #5 clk = 1'b1;//45s
//        #5 clk = 1'b0;
//        #5 clk = 1'b1;//55s
    end
    
    initial begin
        WR = 1'b0;
        #10 WR = 1'b1;
//        #10 WR = 1'b0;//20s
//        #10 WR = 1'b1;//30s
//        #10 WR = 1'b0;//40s
//        #10 WR = 1'b1;//50s
    end
    
    initial begin
        in1 = 32'b00000000000000000000000000010101;
        in2 = 32'b00000000000000000000000000011110;
//        #20 
//        begin
//            in1 = 32'b01000000000000000000000000000000;
//            in2 = 32'b10101000000000000000000000000000;
//        end
//        #20 
//        begin
//            in1 = 32'b01000000000000000000000000000000;
//            in2 = 32'b10110000000000000000000000000000;
//        end
    end
    
    Project_1 I_Project_1(
        .clk(clk),
        .WR(WR),
        .in1(in1),
        .in2(in2),
        .result(result)
    );
endmodule
