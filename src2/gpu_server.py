"""
Thanks: http://stackoverflow.com/a/27412611/266387

Check that len(sys.argv) > 1, otherwise the script crashes on the remote. Also, your bashrc (or shell initial configuration) on the remote must not contain any prints on stdout (echo in bash)
"""
import sys
import os

def main():
    print os.name

if __name__ == '__main__':
    try:
        if sys.argv[1] == 'deploy':
            import paramiko

            # Connect to remote host
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect('remote_hostname_or_IP', username='john', password='secret')

            # Setup sftp connection and transmit this script
            sftp = client.open_sftp()
            sftp.put(__file__, '/tmp/myscript.py')
            sftp.close()

            # Run the transmitted script remotely without args and show its output.
            # SSHClient.exec_command() returns the tuple (stdin,stdout,stderr)
            stdout = client.exec_command('python /tmp/myscript.py')[1]
            for line in stdout:
                # Process each line in the remote output
                print line

            client.close()
            sys.exit(0)
    except IndexError:
        pass

    # No cmd-line args provided, run script normally
    main()