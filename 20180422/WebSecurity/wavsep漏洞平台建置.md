# wavsep
```
WAVSEP 是一個包含漏洞的web應用程式，目的是説明測試web應用漏洞掃描器的功能、品質和準確性。
WAVSEP 收集了很多獨特的包含漏洞的web頁面，用於測試web應用程式掃描器的多種特特性。

https://github.com/sectooladdict/wavsep

https://github.com/sectooladdict/wavsep/wiki/WAVSEP-Features

目前WAVSEP支持的漏洞包括：
Reflected XSS: 66 test cases, implemented in 64 jsp pages (GET & POST
Error Based SQL Injection: 80 test cases, implemented in 76 jsp pages (GET & POST )
Blind SQL Injection: 46 test cases, implemented in 44 jsp pages (GET & POST )
Time Based SQL Injection: 10 test cases, implemented in 10 jsp pages (GET & POST )
```
# 下載wavsep
```
https://code.google.com/archive/p/wavsep/downloads
wavsep-v1.2-war-linux.zip

https://github.com/sectooladdict/wavsep
```

# 安裝wavsep
```
Ubuntu14.04下部署wavsep
https://blog.csdn.net/lesliegail1/article/details/69949041

http://www.freebuf.com/sectool/125940.html

http://www.dayexie.com/detail1930197.html
```
```
一、安裝jdk
1.下載jdk
http://www.oracle.com/technetwork/java/javase/downloads/jdk7-downloads-1880260.html

2.解壓 tar zxvf jdk-7u45-linux-x64.tar.gz

3.解壓後得到 jdk1.7.0_52 將其移至 /opt/java/jdk 中（便於文件管理）

4.設置環境變數
vim ~./bashrc

在文件末尾添加：
export JAVA_HOME=/opt/Java/jdk/jdk1.7.0_52
export CLASSPATH=${JAVA_HOME}/lib
export PATH=${JAVA_HOME}/bin:$PATH

5驗證 輸入 java -version 後，得到 版本資訊即表明安裝成功

二、安裝tomcat
sudo apt-get install tomcat6

三、安裝mysql
sudo apt-get install mysql-server-5.5

四、安裝 wavsep
1.將wasvep.war 放到 /val/lib/tomcat6/wabapps 中

2.創建db文件
sudo mkdir /var/lib/tomcat6/db
sudo chown tomcat6:tomcat6 /var/lib/tomcat6/db/

3.訪問 http://localhost:8080/wavsep/wavsep-install/install.jsp

4.填寫相關資訊

5.安裝完成 訪問：http://localhost:8080/wavsep/
```


# 使用漏洞掃描器測試wavsep

