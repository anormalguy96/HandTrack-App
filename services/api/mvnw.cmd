@echo off
setlocal

set "JAVA_EXE=java"
if not "%JAVA_HOME%" == "" (
  set "JAVA_EXE=%JAVA_HOME%\bin\java.exe"
)

if not exist ".mvn\wrapper\maven-wrapper.jar" (
  echo Downloading Maven Wrapper...
  powershell -Command "Invoke-WebRequest -Uri 'https://repo.maven.apache.org/maven2/org/apache/maven/wrapper/maven-wrapper/3.3.2/maven-wrapper-3.3.2.jar' -OutFile '.mvn/wrapper/maven-wrapper.jar'"
)

"%JAVA_EXE%" -Dmaven.multiModuleProjectDirectory=%~dp0 -cp .mvn\wrapper\maven-wrapper.jar org.apache.maven.wrapper.MavenWrapperMain %*

if ERRORLEVEL 1 (
  echo Maven Wrapper failed.
  exit /b 1
)

endlocal
