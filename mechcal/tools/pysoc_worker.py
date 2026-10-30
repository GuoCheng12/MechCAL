"""Standalone worker in the isolated PySOC interpreter."""

import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np


def main():
    request=json.loads(Path(sys.argv[1]).read_text())
    source,inputs,output=(Path(request[k]) for k in ('source','inputs','output'))
    for name,expected in request['source_hashes'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Numerical implementation changed: '+name)
    sys.path.insert(0,str(source))
    import pysoc
    from pysoc.io.dftb_plus import DFTB_plus_parser
    from pysoc.io.soc_td import Soc_td
    if pysoc.version!='2.3.0':
        raise ValueError('Unvalidated PySOC version')
    canonical_path=inputs/'canonical.json'
    canonical=json.loads(canonical_path.read_text())
    molsoc=DFTB_plus_parser.from_output_files(inputs/'geometry.xyz',
        requested_singlets=[1,2,3],requested_triplets=[1,2,3])
    molsoc.parse()
    for key,value in canonical.items():
        actual=getattr(molsoc,key)
        if key=='geometry':
            if [r[0] for r in actual]!=[r[0] for r in value]:
                raise ValueError('Legacy parser changes atom order')
            actual,value=[r[1:] for r in actual],[r[1:] for r in value]
        a,b=np.asarray(actual,dtype=float),np.asarray(value,dtype=float)
        if a.shape!=b.shape or not np.allclose(a,b,atol=1e-10,rtol=0):
            raise ValueError('Legacy parsing changes canonical '+key)
    if not molsoc.check_zeff():
        raise ValueError('Requested ZEFF unsupported; no operator substitution')
    output.mkdir(exist_ok=False)
    keywords=['ANG','DIP','TDB','ZEFF']
    molsoc.prepare(keywords,1.0,output)
    molsoc.run()
    soc=Soc_td(molsoc)
    soc.prepare(keywords,True,1e-5)
    soc.run()
    rows=[]
    for row in soc.table[1:]:
        rows.append(dict(singlet=int(re.fullmatch(r'S\((\d+)\)',row[0])[1]),
            triplet=int(re.fullmatch(r'T\((\d+)\)',row[1])[1]),
            rss_cm_inverse=row[2],components_cm_inverse=row[3:]))
    values={(r['singlet'],r['triplet']):r for r in rows}
    seen=set()
    for line in (output/'soc_out.dat').read_text().splitlines():
        match=re.fullmatch(r'sum_soc, <S\((\d+)\)\|Hso\|T\((\d+)\),1,0,-1> \(cm-1\):\s*(.*)',line)
        if not match:
            raise ValueError('Unknown SOC output line')
        key=(int(match[1]),int(match[2]))
        if key in seen or key not in values:
            raise ValueError('Duplicated or unknown SOC pair')
        seen.add(key)
        printed=np.asarray([float(v) for v in match[3].split()])
        expected=np.asarray([values[key]['rss_cm_inverse'],*values[key]['components_cm_inverse']])
        if printed.shape!=(4,) or not np.isfinite(printed).all() or np.max(np.abs(printed-expected))>3e-5:
            raise ValueError('Reported SOC total/components disagree')
    if len(rows)!=12 or seen!={(s,t) for s in range(4) for t in range(1,4)}:
        raise ValueError('Incomplete SOC calculation')
    result=dict(schema_version='pysoc_zeff_v1',status='success',pysoc_version=pysoc.version,
        method='TD-DFTB/mio-1-1 + PySOC ZEFF',unit='cm^-1',operator='ZEFF',scale=1.0,ci_threshold=1e-5,
        canonical_input_sha256=hashlib.sha256(canonical_path.read_bytes()).hexdigest(),
        source_hashes=request['source_hashes'],couplings=rows)
    Path(sys.argv[2]).write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
