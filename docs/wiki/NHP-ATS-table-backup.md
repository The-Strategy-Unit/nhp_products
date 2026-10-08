# NHP ATS table backups

The [nhp_ats_backup](https://github.com/The-Strategy-Unit/nhp_ats_backup)
repository provides automated backups of the NHP Azure Table Storage (ATS)
table. It takes a full JSON snapshot every day at 02:00 UTC and stores it in
Azure Blob Storage.

The seven newest snapshots are retained, along with six monthly snapshots. The
backup process also records the latest successful snapshot in `status.json`.

# Restoring a table

A restore replaces the whole target table: all existing entities are deleted
before the snapshot is loaded. Particular care is required when selecting the 
target table.

For the current [backup](https://github.com/The-Strategy-Unit/nhp_ats_backup#backup) and
[restore](https://github.com/The-Strategy-Unit/nhp_ats_backup#restore) instructions, 
see the repository README.
