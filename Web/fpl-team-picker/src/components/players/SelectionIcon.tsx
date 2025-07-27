import { Player, SelectedSquad } from "../../helpers/api";

function SelectionIcon({ player, team }: { player: Player, team: SelectedSquad | null }) {

  let badgeText = '';
  let cls = '';
  const inXi = team?.startingXi!.some(p => p.player?.id === player.id);
  if (inXi) {
    badgeText = 'XI';
    cls = 'bg-primary text-primary-foreground'
  }
  const onBench = team?.bench!.some(p => p.player?.id === player.id);
  if (onBench) {
    badgeText = 'B';
    cls = 'border border-primary text-primary bg-primary/10'
  }

  if (!badgeText) {
    return <></>;
  }
  return <div className={'text-xs rounded-sm font-mono mx-1 px-2 ' + cls}>{badgeText}</div>
}

export default SelectionIcon;