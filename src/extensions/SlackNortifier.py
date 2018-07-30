import json
import chainer
from chainer.training.extensions import log_report as log_report_module
import requests

def post2slack(text, username, channel, slack_url):
    payload = {
        "text": text,
        "username": username,
        "channel": channel,
        "icon_emoji": "ghost:"
    }

    res = requests.post(slack_url, data=json.dumps(payload))
    if res.status_code != 200:
        res.raise_for_status()

class SlackNortifier(chainer.training.Extension):
    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self,
            entries,
            slack_url,
            log_report='LogReport',
            username='',
            channel='',
    ):
        self._entries = entries
        self._log_report = log_report
        self._log_len = 0
        self.username = username
        self.channel = channel
        self.slack_url = slack_url

        entry_widths = [max(10, len(s)) for s in entries]
        header = '  '.join(('{:%d}' % w for w in entry_widths)).format(*entries) + '\n'
        self._header = header  # printed at the first call

        templates = []
        for entry, w in zip(entries, entry_widths):
            templates.append((entry, '{:<%dg}  ' % w, ' ' * (w + 2)))
        self._templates = templates

    def __call__(self, trainer):
        if self._header:
            self._post2slack(self._header)
            self._header = None

        log_report = self._log_report
        if isinstance(log_report, str):
            log_report = trainer.get_extension(log_report)
        elif isinstance(log_report, log_report_module.LogReport):
            log_report(trainer)
        else:
            raise TypeError('log report has a wrong type %s' %
                            type(log_report))

        log = log_report.log
        log_len = self._log_len
        while len(log) > log_len:
            self._observation_post2slack(log[log_len])
            log_len += 1
        self._log_len = log_len

    def _post2slack(self, text):
        payload = {
            "text": text,
            "username": self.username,
            "channel": self.channel,
            "icon_emoji": "ghost:"
        }

        res = requests.post(self.slack_url, data=json.dumps(payload))
        if res.status_code != 200:
            res.raise_for_status()

    def serialize(self, serializer):
        log_report = self._log_report
        if isinstance(log_report, log_report_module.LogReport):
            log_report.serialize(serializer['_log_report'])

    def _observation_post2slack(self, observation):
        text = ''
        for entry, template, empty in self._templates:
            if entry in observation:
                text += template.format(observation[entry])
            else:
                text += empty

        self._post2slack(text)
