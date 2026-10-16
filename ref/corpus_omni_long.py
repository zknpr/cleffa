"""Fixed 64000-token Omni context fixture; lengths verified by the pinned encoder."""

def build():
    return [dict(id='ol000',model='clef-omni',state=' alpha'*63857+' FINAL: billing outage.',
                 questions={'outage':dict(type='noul',instructions='Is a service down?')})]
