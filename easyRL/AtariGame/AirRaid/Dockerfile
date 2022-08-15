FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.7-cuda11.0-py3
RUN pip install --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple gym gym[atari] ale-py autorom matplotlib seaborn && \
    AutoROM --accept-license && \
    pip uninstall ale-py gym[atari] -y && \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ale-py gym[atari] && \
    echo "python easyRL/AtariGame/AirRaid/task.py" > /opt/ForCommit/run.sh && \
    chmod +x /opt/ForCommit/run.sh
COPY ForCommit /opt/ForCommit
WORKDIR /opt/ForCommit
ENV PYTHONPATH /opt/ForCommit:$PYTHONPATH
CMD ["/bin/sh","run.sh"]