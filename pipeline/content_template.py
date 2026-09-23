"""
content_template.py

Removes `ansible-galaxy init` TEMPLATE TEXT from a role's content before it is
vectorised. The structural counterpart already exists as filter_boilerplate()
in find_clones_at_scale_v3.py, which strips generated PATHS from structure
similarity; this does the same job for the words inside those files.

WHY
---
`ansible-galaxy init` writes 2,927 bytes across 8 files. README.md (1,328) and
meta/main.yml (1,209) are 86% of it, and both are mostly English prose an author
never has to touch. Measured over a 4,000-role sample of the Dec 2023 corpus:

    100.0%  galaxy_info:                  46.5%  License
     91.5%  platforms:                    45.1%  Role Variables
     89.0%  versions:                     43.7%  Example Playbook
     76.2%  dependencies: []              40.1%  Requirements
     67.2%  galaxy_tags:                  38.1%  Author Information

About 30% of roles never edited the scaffold comments ("# tasks file for ...",
"# vars file for ..."), and ~9% still carry the whole unedited README.

WHERE THE LINE LIST COMES FROM
------------------------------
Two sources, because one is not enough:

  1. `ansible-galaxy init` run live, if the binary is present. This is ground
     truth for what the CURRENT generator emits.
  2. LEGACY_TEMPLATE below -- what OLDER generators emitted. The corpus was
     collected December 2023 and its roles were created over many years. The
     "#SPDX-License-Identifier: MIT-0" header that today's generator writes
     appears in 0.0% of the corpus, which is the proof that matching against
     the current template alone would miss almost everything.

The generator embeds the role's own name ("# tasks file for geerlingguy.mysql"),
so strip_template() normalises that to <ROLE> before comparing.

MEASURED EFFECT (corpus-wide, 14,582 labelled pairs)
----------------------------------------------------
    mode          F1       AUC      d fork   d non-fork
    none        96.14%   0.9929      +0.00       +0.00
    galaxy_init 96.17%   0.9931      -0.08       -0.32

The direction is right -- unrelated pairs fall four times harder than true forks
-- but the size is 0.03 F1 points. TF-IDF's inverse-document-frequency term
already discounts text this common: "galaxy_info:" is in 100% of roles, so its
weight is already near zero. Structure similarity (Jaccard) has no such
mechanism, which is why the equivalent filter matters far more there.

Kept opt-in and OFF by default so every existing result stays reproducible.
"""

import os
import re
import subprocess
import tempfile

# Emitted by older `ansible-galaxy init` versions and still present throughout a
# Dec 2023 corpus. Includes the README skeleton headings and their underlines,
# which survive even when the author rewrote the prose around them.
LEGACY_TEMPLATE = [
    # meta/main.yml
    "galaxy_info:", "platforms:", "versions:", "categories:",
    "galaxy_tags: []", "dependencies: []",
    "author: your name", "description: your role description",
    "company: your company (optional)",
    "license: BSD", "license: license (GPLv2, CC-BY, etc)",
    "license: license (GPL-2.0-or-later, MIT, etc)",
    "min_ansible_version: 1.2", "min_ansible_version: 2.2",
    "min_ansible_version: 2.4", "min_ansible_version: 2.9",
    "# platforms is a list of platforms, and each platform has a name and a list of versions.",
    "# List tags for your role here, one per line. A tag is a keyword that describes",
    "# and categorizes the role. Users find roles by searching for tags. Be sure to",
    "# remove the '[]' above, if you add tags to this list.",
    "# NOTE: A tag is limited to a single word comprised of alphanumeric characters.",
    "#       Maximum 20 tags per role.",
    "# List your role dependencies here, one per line. Be sure to remove the '[]' above,",
    "# if you add dependencies to this list.",
    "# If the issue tracker for your role is not on github, uncomment the",
    "# next line and provide a value",
    "# issue_tracker_url: http://example.com/issue/tracker",
    "# Choose a valid license ID from https://spdx.org - some suggested licenses:",
    "# If this a Container Enabled role, provide the minimum Ansible Container version.",
    # README.md skeleton -- headings and their underlines
    "Role Name", "=========",
    "Requirements", "------------",
    "Role Variables", "--------------",
    "Dependencies",
    "Example Playbook", "----------------",
    "License", "-------",
    "Author Information", "------------------",
    "A brief description of the role goes here.",
    "Any pre-requisites that may not be covered by Ansible itself or the role should be mentioned here. For instance, if the role uses the EC2 module, it may be a good idea to mention in this section that the boto package is required.",
    "A description of the settable variables for this role should go here, including any variables that are in defaults/main.yml, vars/main.yml, and any variables that can/should be set via parameters to the role. Any variables that are read from other roles and/or the global scope (ie. hostvars, group vars, etc.) should be mentioned here as well.",
    "A list of other roles hosted on Galaxy should go here, plus any details in regards to parameters that may need to be set for other roles, or variables that are used from other roles.",
    "Including an example of how to use your role (for instance, with variables passed in as parameters) is always nice for users too:",
    "An optional section for the role authors to include contact information, or a website (HTML is not allowed).",
    "- hosts: servers", "roles:", "- { role: username.rolename, x: 42 }",
    "BSD",
    # tests/
    "- hosts: localhost", "remote_user: root", "localhost",
    # the per-file comments the generator writes into every empty file
    "# tasks file for <ROLE>", "# defaults file for <ROLE>",
    "# handlers file for <ROLE>", "# vars file for <ROLE>",
]

_CACHE = None


def template_lines(verbose=False):
    """The set of lines to strip. Runs `ansible-galaxy init` when available and
    unions its output with LEGACY_TEMPLATE. Cached per process."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    lines, src = set(), "legacy list only"
    try:
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(["ansible-galaxy", "init", "tmpl_role"], cwd=td,
                           capture_output=True, timeout=60, check=True)
            root = os.path.join(td, "tmpl_role")
            for dp, _, fn in os.walk(root):
                for f in fn:
                    with open(os.path.join(dp, f), encoding="utf-8", errors="ignore") as fh:
                        for line in fh:
                            n = re.sub(r"\btmpl_role\b", "<ROLE>", line.strip())
                            if len(n) > 3:
                                lines.add(n)
            src = "ansible-galaxy init + legacy list"
    except Exception:
        pass
    lines.update(LEGACY_TEMPLATE)
    if verbose:
        print(f"    content boilerplate: {len(lines)} template lines ({src})")
    _CACHE = lines
    return lines


def strip_template(text, rolename="", lines=None):
    """Drop generated template lines from a role's concatenated text.

    rolename is the part after the dot ("mysql" in "geerlingguy.mysql"); the
    generator writes it into the per-file comments, so it is normalised to
    <ROLE> before matching. Short lines (<= 3 chars, e.g. '---') are always kept:
    they are syntax, not prose, and removing them would change nothing about
    similarity while making the filter harder to defend.
    """
    drop = lines if lines is not None else template_lines()
    pat = re.escape(rolename) if rolename else None
    keep = []
    for raw in text.splitlines():
        s = raw.strip()
        if len(s) <= 3:
            keep.append(raw)
            continue
        n = re.sub(pat, "<ROLE>", s) if pat else s
        if n in drop or s in drop:
            continue
        keep.append(raw)
    return "\n".join(keep)
